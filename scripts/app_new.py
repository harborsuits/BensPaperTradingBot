"""
BensBot Autonomous Trading Dashboard
-----------------------------------
A streamlined UI focused exclusively on the autonomous trading pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time
import traceback
import uuid
import logging

# Constants for feature availability
AI_CHAT_AVAILABLE = True  # Set to True to enable the BenBot AI chat assistant
CHART_LIBRARY_AVAILABLE = True  # Set to True to enable Plotly charts
PORTFOLIO_DATA_AVAILABLE = True  # Set to True to enable real portfolio data

# Import UI components
from ui.ai_chat import render_ai_chat_widget

# External components
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    CHART_LIBRARY_AVAILABLE = False
    st.warning("Plotly not installed. Some visualizations may not be available.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dashboard")

# Import BenBot Assistant if available
try:
    from trading_bot.assistant.benbot_assistant import BenBotAssistant
    BENBOT_AVAILABLE = True
    logger.info("BenBot Assistant module found and imported successfully")
except ImportError as e:
    BENBOT_AVAILABLE = False
    logger.warning(f"BenBot Assistant module not found, will use fallback responses: {e}")

# Try to import the real trading bot components
try:
    # Import the Orchestrator for autonomous trading
    from trading_bot.core.main_orchestrator import MainOrchestrator
    
    # Import the data manager for market data
    from trading_bot.data.data_manager import DataManager
    
    # Import the BacktestManager for backtest operations
    from trading_bot.backtesting.backtest_manager import BacktestManager
    
    # Import the strategy factory
    from trading_bot.strategies.strategy_factory import StrategyFactory
    
    # Import the portfolio manager
    from trading_bot.portfolio.portfolio_manager import PortfolioManager
    
    # Import news service
    from trading_bot.services.news_service import NewsService
    
    # Import alert manager
    from trading_bot.alerts.alert_manager import AlertManager
    
    # Set flag for real components
    REAL_COMPONENTS_AVAILABLE = True
    logger.info("Successfully imported trading_bot components")
    
except ImportError as e:
    # Log the error but continue with mock components
    logger.warning(f"Error importing trading_bot components: {e}")
    logger.warning("Using mock data for dashboard. Some features will be limited.")
    REAL_COMPONENTS_AVAILABLE = False

# Set page config - professional dark theme
st.set_page_config(
    page_title="BensBot Autonomous Trading",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- SHARED STATE --------------------
if "active_symbol" not in st.session_state:
    st.session_state.active_symbol = "SPY"  # Default symbol

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Dashboard"

if "last_pipeline_run" not in st.session_state:
    st.session_state.last_pipeline_run = None
    
# Initialize Benbot chat state if not present
if "chat_shown" not in st.session_state:
    st.session_state.chat_shown = False
    
if "benbot_initialized" not in st.session_state:
    st.session_state.benbot_initialized = False

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    /* Global styling */
    .stApp {
        background-color: #0f172a;
        color: white;
    }
    
    /* Card styling */
    .card {
        background-color: #1d3b53;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
        color: white;
    }
    
    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        color: white;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 0.5rem;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-success {
        background-color: #4caf50;
        color: white;
    }
    
    .badge-warning {
        background-color: #ff9800;
        color: white;
    }
    
    .badge-danger {
        background-color: #f44336;
        color: white;
    }
    
    /* Button styling */
    .button {
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        font-weight: 600;
        cursor: pointer;
        border: none;
        display: inline-block;
        text-align: center;
    }
    
    .button-primary {
        background-color: #2196f3;
        color: white;
    }
    
    .button-danger {
        background-color: #f44336;
        color: white;
    }
    
    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-active {
        background-color: #4caf50;
    }
    
    .status-warning {
        background-color: #ff9800;
    }
    
    .status-inactive {
        background-color: #f44336;
    }
    
    /* Custom tabs */
    .custom-tabs {
        display: flex;
        border-bottom: 1px solid rgba(255,255,255,0.2);
        margin-bottom: 1rem;
    }
    
    .custom-tab {
        padding: 0.8rem 1.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
        border-bottom: 2px solid transparent;
        color: rgba(255,255,255,0.7);
    }
    
    .custom-tab.active {
        border-bottom: 2px solid #2196f3;
        color: white;
        font-weight: 600;
    }
    
    /* Header */
    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    /* Utils */
    .text-success { color: #4caf50; }
    .text-warning { color: #ff9800; }
    .text-danger { color: #f44336; }
    .text-muted { color: rgba(255,255,255,0.6); }
</style>
""", unsafe_allow_html=True)

# Initialize services
try:
    # Initialize news service
    news_service = NewsService(API_KEYS)
    
    # Initialize other components if available
    if REAL_COMPONENTS_AVAILABLE:
        data_hub = CentralDataHub(API_KEYS)
        alert_manager = AlertManager()
        orchestrator = AutonomousOrchestrator(data_hub=data_hub)
        print("Initialized full autonomous trading stack")
    else:
        data_hub = None
        alert_manager = None
        orchestrator = None
        print("Initialized with limited services - some features will use mock data")
    
    print("Services initialized successfully")
except Exception as e:
    st.error(f"Failed to initialize services: {e}")
    news_service = None
    data_hub = None
    alert_manager = None
    orchestrator = None

# Initialize component instances if real components are available
try:
    if REAL_COMPONENTS_AVAILABLE:
        # Create instances of real components
        orchestrator = MainOrchestrator()
        data_manager = DataManager()
        backtest_manager = BacktestManager()
        strategy_factory = StrategyFactory()
        portfolio_manager = PortfolioManager()
        news_service = NewsService()
        alert_manager = AlertManager()
        
        logger.info("Successfully initialized trading_bot components")
    else:
        # Use mock instances
        orchestrator = None
        data_manager = None
        backtest_manager = None
        strategy_factory = None
        portfolio_manager = None
        news_service = None
        alert_manager = None
        
        logger.info("Using mock components for dashboard")
except Exception as e:
    # Log the error but continue with None components
    logger.error(f"Error initializing trading_bot components: {e}")
    orchestrator = None
    data_manager = None
    backtest_manager = None
    strategy_factory = None
    portfolio_manager = None
    news_service = None
    alert_manager = None

# Initialize BenBot Assistant with direct orchestrator integration
benbot_assistant = None
if BENBOT_AVAILABLE:
    try:
        # Create a dictionary with all available trading components for the assistant to use
        trading_context = {
            "orchestrator": orchestrator,
            "data_manager": data_manager,
            "backtest_manager": backtest_manager,
            "strategy_factory": strategy_factory,
            "portfolio_manager": portfolio_manager,
            "news_service": news_service,
            "alert_manager": alert_manager
        }
        
        # Log availability of components
        logger.info(f"Orchestrator available: {orchestrator is not None}")
        logger.info(f"Data Manager available: {data_manager is not None}")
        
        # Initialize the assistant with trading context and paths
        benbot_assistant = BenBotAssistant(
            data_manager=data_manager,
            dashboard_interface=None,
            data_dir="data",
            results_dir="results",
            models_dir="models"
        )
        
        # Store trading context in the assistant for direct access to trading components
        benbot_assistant.trading_context = trading_context
        
        logger.info("BenBot Assistant initialized and connected to trading orchestrator")
        
        # Store in session state
        st.session_state.benbot_assistant = benbot_assistant
        st.session_state.benbot_initialized = True
        
        # Enable the chat by default since we have a working assistant
        st.session_state.chat_shown = True
    except Exception as e:
        logger.error(f"Failed to initialize BenBot Assistant: {str(e)}\n{traceback.format_exc()}")
        BENBOT_AVAILABLE = False
        st.session_state.benbot_initialized = False

# -------------------- AUTONOMOUS TAB COMPONENTS --------------------

def opportunity_discovery_section(discovered_opportunities):
    """Display discovered opportunities before approval.
    
    Args:
        discovered_opportunities (list): List of opportunities found by system
    """
    # More professional header with count badge
    discovery_count = len(discovered_opportunities) if discovered_opportunities else 0
    st.markdown(f"<h2>Opportunity Discovery <span style='font-size: 1rem; color: rgba(255,255,255,0.7); font-weight: normal; background-color: rgba(33,150,243,0.2); padding: 0.2rem 0.5rem; border-radius: 1rem;'>{discovery_count} matches</span></h2>", unsafe_allow_html=True)
    st.markdown("<p style='margin-bottom: 1.5rem;'>Market symbols and strategies identified by our AI models that match your criteria</p>", unsafe_allow_html=True)

    if not discovered_opportunities:
        st.info("No new opportunities discovered in the latest scan. Next scan: [scheduled time]")
        return
    
    # Create a table for opportunities
    cols = ['Symbol', 'Discovery Time', 'Source', 'Score', 'Key Evidence', 'Status', 'Actions']  
    data = [
        [d['symbol'], d['timestamp'], d['source'], f"{d['score']:.2f}", d['evidence'][:50] + "...", d['status'], "View"] 
        for d in discovered_opportunities
    ]
    
    df = pd.DataFrame(data, columns=cols)
    
    # Add color highlighting based on score
    def highlight_score(val):
        if isinstance(val, float) or (isinstance(val, str) and val.replace('.', '', 1).isdigit()):
            score = float(val)
            if score >= 0.8:
                return 'background-color: #4CAF50; color: white'
            elif score >= 0.6:
                return 'background-color: #FFC107; color: black'
            else:
                return 'background-color: #F44336; color: white'
    
    # Apply styling and display table
    styled_df = df.style.applymap(highlight_score, subset=['Score'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Add button to view details of selected opportunity
    if st.button("View Details of Selected Opportunity", key="view_opportunity_details"):
        st.session_state.show_opportunity_details = True

# This function has been replaced by render_autonomous_tab
# def autonomous_tab():
#     """Create the autonomous pipeline control tab."""
#     # Simple professional header
#     st.markdown("<h1 style='font-size: 2rem; margin-bottom: 1.5rem;'>Autonomous Trading System</h1>", unsafe_allow_html=True)

def backtest_results_section(backtests):
    """Display backtest results for strategies being evaluated.
    
    Args:
        backtests (list): List of backtest results
    """
    # More professional header with count badge
    backtest_count = len(backtests) if backtests else 0
    st.markdown(f"<h2>Backtest Results <span style='font-size: 1rem; color: rgba(255,255,255,0.7); font-weight: normal; background-color: rgba(33,150,243,0.2); padding: 0.2rem 0.5rem; border-radius: 1rem;'>{backtest_count} strategies</span></h2>", unsafe_allow_html=True)
    st.markdown("<p style='margin-bottom: 1.5rem;'>Backtest results for strategies being evaluated</p>", unsafe_allow_html=True)

    if not backtests:
        st.info("No active backtests currently running. Strategies will appear here when testing begins.")
        return
    
    # Create tabs for each backtest
    backtest_tabs = st.tabs([f"{b['symbol']} - {b['strategy']}" for b in backtests])
    
    for i, tab in enumerate(backtest_tabs):
        with tab:
            backtest = backtests[i]
            
            # Display backtest summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{backtest['return']:+.2%}")
            with col2:
                st.metric("Sharpe Ratio", f"{backtest['sharpe']:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{backtest['drawdown']:.2%}")
            with col4:
                st.metric("Win Rate", f"{backtest['win_rate']:.1%}")
            
            # Display equity curve
            st.markdown("#### Equity Curve vs Benchmark")
            if PLOTLY_AVAILABLE:
                # Use plotly if available
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=backtest['dates'],
                    y=backtest['equity_curve'],
                    mode='lines',
                    name='Strategy',
                    line=dict(color='#2196F3', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=backtest['dates'],
                    y=backtest['benchmark'],
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#757575', width=1, dash='dash')
                ))
                fig.update_layout(
                    title=f"{backtest['strategy']} on {backtest['symbol']}",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value",
                    template="plotly_dark",
                    plot_bgcolor="#1d3b53",
                    paper_bgcolor="#1d3b53",
                    font=dict(color="white"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to text
                st.info("Plotly not available for charting. Install plotly for interactive charts.")
                st.write(f"Strategy final return: {backtest['return']:.2%}")
            
            # Display optimization history
            st.markdown("#### Parameter Optimization")
            if 'optimization_history' in backtest and backtest['optimization_history']:
                opt_df = pd.DataFrame(backtest['optimization_history'])
                st.dataframe(opt_df, use_container_width=True)
                
                # Show improvement from optimization
                if len(backtest['optimization_history']) > 1:
                    initial = backtest['optimization_history'][0]['performance']
                    final = backtest['optimization_history'][-1]['performance']
                    improvement = ((final - initial) / abs(initial)) * 100 if initial != 0 else float('inf')
                    st.success(f"🚀 AI optimization improved performance by {improvement:.1f}%")
            else:
                st.info("No optimization history available for this strategy.")

def approval_queue_section(approved_opportunities):
    """Display strategies ready for approval and execution.
    
    Args:
        approved_opportunities (list): List of validated opportunities ready for approval
    """
    # More professional header with count and description
    opportunity_count = len(approved_opportunities) if approved_opportunities else 0
    st.markdown(f"<h2>Approval Queue <span style='font-size: 1rem; color: rgba(255,255,255,0.7); font-weight: normal;'>({opportunity_count} opportunities)</span></h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; margin-bottom: 1.5rem;'>Validated trading opportunities ready for your final review and approval</p>", unsafe_allow_html=True)
    
    if not approved_opportunities:
        st.info("No opportunities have passed validation yet. Check back soon or adjust discovery parameters.")
        return
    
    # Ensure each opportunity has an 'id' key, generating a UUID if missing
    for opp in approved_opportunities:
        if 'id' not in opp:
            opp['id'] = f"opp_{uuid.uuid4().hex[:8]}"
    
    # Sort opportunities by a combined score of confidence and expected return
    # This ensures the highest quality opportunities appear at the top
    if len(approved_opportunities) > 1:
        for opp in approved_opportunities:
            if 'quality_score' not in opp:
                # Calculate a combined quality score (70% confidence, 30% expected return)
                confidence_component = opp['confidence'] * 0.7
                return_component = min(max(opp['expected_return'] / 0.3, 0), 1) * 0.3  # Normalize expected return
                opp['quality_score'] = confidence_component + return_component
        
        # Sort by quality score (highest first)
        approved_opportunities = sorted(approved_opportunities, key=lambda x: x.get('quality_score', 0), reverse=True)
    
    # Create a container for each opportunity
    for opp in approved_opportunities:
        # Get or generate status and waiting time
        status = opp.get('status', 'Awaiting Approval')
        submission_time = opp.get('submission_time', datetime.now() - timedelta(hours=np.random.randint(1, 24)))
        if isinstance(submission_time, str):
            try:
                submission_time = datetime.strptime(submission_time, "%Y-%m-%d %H:%M:%S")
            except:
                submission_time = datetime.now() - timedelta(hours=np.random.randint(1, 24))
        
        waiting_time = datetime.now() - submission_time
        waiting_hours = waiting_time.total_seconds() / 3600
        
        # Determine status color
        status_color = "#FFC107"  # Default yellow
        if status == "Ready":
            status_color = "#4CAF50"  # Green
        elif "urgent" in status.lower():
            status_color = "#F44336"  # Red
        
        # Format price targets and strategy parameters
        entry_price = opp.get('entry_price', opp.get('current_price', 0) * 0.98)
        stop_loss = opp.get('stop_loss', entry_price * 0.95)
        profit_target = opp.get('profit_target', entry_price * 1.15)
        
        # Calculate percentages
        stop_loss_pct = (stop_loss - entry_price) / entry_price
        profit_target_pct = (profit_target - entry_price) / entry_price
        
        # Format asset type and position sizing with better presentation
        asset_type = opp.get('asset_type', 'stock')
        
        # Get a friendly display name for the asset type
        asset_type_display = {
            'stock': 'Stock',
            'crypto': 'Crypto',
            'forex': 'Forex',
            'options': 'Options',
            'futures': 'Futures'
        }.get(asset_type.lower(), asset_type)
        
        # Format position size
        position_size = opp.get('position_size', '$10,000')
        if isinstance(position_size, (int, float)):
            position_size = f"${position_size:,.0f}"
        
        # Get additional exit conditions
        exit_conditions = opp.get('exit_conditions', 'Time-based exit after 15 trading days if neither target nor stop is hit')
        
        # Create a Streamlit card with clear visual separation
        with st.container():
            # Add improved card styling with CSS
            st.markdown("""
            <style>
            .opportunity-card {
                background-color: rgba(33,150,243,0.1); 
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 30px;
                border: 1px solid rgba(33,150,243,0.3);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            .card-header {
                margin: -20px -20px 15px -20px;
                padding: 15px 20px;
                background-color: rgba(33,150,243,0.2);
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }
            .status-badge {
                display: inline-block;
                padding: 5px 12px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: bold;
                margin-right: 10px;
            }
            .metrics-container {
                background-color: rgba(255,255,255,0.05); 
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
            }
            .entry-exit-container {
                background-color: rgba(0,0,0,0.1);
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
            }
            .strategy-params {
                background-color: rgba(255,255,255,0.03);
                padding: 10px;
                border-radius: 5px;
                border-left: 3px solid #2196F3;
                margin: 15px 0;
            }
            .evidence-container {
                border-top: 1px solid rgba(255,255,255,0.1);
                padding-top: 15px;
                margin-top: 10px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create a card with clear visual separation
            st.markdown("<div class='opportunity-card'>", unsafe_allow_html=True)
            
            # Create asset type badge color
            asset_badge_colors = {
                'stock': '#4CAF50',    # Green
                'crypto': '#FF9800',   # Orange
                'forex': '#2196F3',    # Blue
                'options': '#9C27B0',  # Purple
                'futures': '#F44336'   # Red
            }
            asset_badge_color = asset_badge_colors.get(asset_type.lower(), '#607D8B')
            
            # Get a cleaner strategy display name by removing the asset prefix
            strategy_display = opp.get('strategy', f"{asset_type.lower()}_strategy")
            if '_' in strategy_display:
                # Try to remove the asset type prefix (e.g., 'stock_momentum' -> 'Momentum')
                strategy_parts = strategy_display.split('_')
                if strategy_parts[0].lower() in ['stock', 'crypto', 'forex', 'options', 'futures']:
                    strategy_display = ' '.join(part.title() for part in strategy_parts[1:])
            else:
                # If no underscore or not properly formatted, create a default display name
                strategy_display = strategy_display.replace("_", " ").title()
            
            # Header row with symbol, asset type badge, and status in a colored header section
            st.markdown(f"""<div class='card-header'><div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h3 style='margin: 0; font-size: 1.4rem;'>{opp.get('symbol', f"Opp_{opp.get('id', 'Unknown')}")}</h3>
                    <div style='display: flex; align-items: center; margin-top: 5px;'>
                        <span style='background-color: {asset_badge_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; margin-right: 8px;'>{asset_type_display}</span>
                        <span style='font-size: 0.9rem;'>{strategy_display}</span>
                    </div>
                </div>
                <div>
                    <span class='status-badge' style='background-color: {status_color}; color: white;'>{status}</span>
                    <span style='color: rgba(255,255,255,0.7); font-size: 0.8rem;'>Waiting: {waiting_hours:.1f} hours</span>
                </div>
            </div></div>""", unsafe_allow_html=True)
            
            # Metrics section
            st.markdown("<div class='metrics-container'></div>", unsafe_allow_html=True)
            metric_cols = st.columns(5)
            with metric_cols[0]:
                # Use .get() with default values to prevent KeyErrors
                expected_return = opp.get('expected_return', 0.0)
                st.markdown(f"**Expected Return:**<br>{expected_return:+.2%}", unsafe_allow_html=True)
            with metric_cols[1]:
                sharpe = opp.get('sharpe', 0.0)
                st.markdown(f"**Sharpe:**<br>{sharpe:.2f}", unsafe_allow_html=True)
            with metric_cols[2]:
                confidence = opp.get('confidence', 0.0)
                st.markdown(f"**Confidence:**<br>{confidence:.0%}", unsafe_allow_html=True)
            with metric_cols[3]:
                # Format position size with appropriate currency/unit symbol based on asset type
                position_label = "Position Size:"
                if asset_type.lower() == 'forex':
                    position_display = f"{position_size} notional"
                elif asset_type.lower() == 'crypto':
                    position_display = position_size
                elif asset_type.lower() == 'options':
                    position_display = f"{position_size} premium"
                else:
                    position_display = position_size
                st.markdown(f"**{position_label}**<br>{position_display}", unsafe_allow_html=True)
            with metric_cols[4]:
                # Display discovery reason if available
                discovery_reason = opp.get('discovery_reason', 'Strategy match')
                st.markdown(f"**Discovery:**<br>{discovery_reason}", unsafe_allow_html=True)
            
            # Entry/Exit targets with asset-specific formatting
            st.markdown("<div class='entry-exit-container'></div>", unsafe_allow_html=True)
            entry_cols = st.columns(3)
            
            # Get entry/exit values with safe defaults
            entry_price = opp.get('entry_price', 0.0)
            stop_loss = opp.get('stop_loss', 0.0)
            profit_target = opp.get('profit_target', 0.0)
            
            # Calculate percentages safely (avoid division by zero)
            if entry_price > 0:
                stop_loss_pct = (stop_loss - entry_price) / entry_price
                profit_target_pct = (profit_target - entry_price) / entry_price
            else:
                stop_loss_pct = 0.0
                profit_target_pct = 0.0
            
            # Format price displays based on asset type
            if asset_type.lower() == 'forex':
                # Format forex prices with more decimals and no $ sign
                with entry_cols[0]:
                    st.markdown(f"<span style='color: #2196F3; font-weight: bold;'>Entry:</span> {entry_price:.4f}", unsafe_allow_html=True)
                with entry_cols[1]:
                    st.markdown(f"<span style='color: #F44336; font-weight: bold;'>Stop-Loss:</span> {stop_loss:.4f} ({stop_loss_pct:+.1%})", unsafe_allow_html=True)
                with entry_cols[2]:
                    st.markdown(f"<span style='color: #4CAF50; font-weight: bold;'>Target:</span> {profit_target:.4f} ({profit_target_pct:+.1%})", unsafe_allow_html=True)
            elif asset_type.lower() == 'crypto':
                # Format crypto with appropriate precision based on price magnitude
                if entry_price > 1000:
                    # Bitcoin-like
                    with entry_cols[0]:
                        st.markdown(f"<span style='color: #2196F3; font-weight: bold;'>Entry:</span> ${entry_price:.2f}", unsafe_allow_html=True)
                    with entry_cols[1]:
                        st.markdown(f"<span style='color: #F44336; font-weight: bold;'>Stop-Loss:</span> ${stop_loss:.2f} ({stop_loss_pct:+.1%})", unsafe_allow_html=True)
                    with entry_cols[2]:
                        st.markdown(f"<span style='color: #4CAF50; font-weight: bold;'>Target:</span> ${profit_target:.2f} ({profit_target_pct:+.1%})", unsafe_allow_html=True)
                else:
                    # Altcoin-like with more precision
                    with entry_cols[0]:
                        st.markdown(f"<span style='color: #2196F3; font-weight: bold;'>Entry:</span> ${entry_price:.4f}", unsafe_allow_html=True)
                    with entry_cols[1]:
                        st.markdown(f"<span style='color: #F44336; font-weight: bold;'>Stop-Loss:</span> ${stop_loss:.4f} ({stop_loss_pct:+.1%})", unsafe_allow_html=True)
                    with entry_cols[2]:
                        st.markdown(f"<span style='color: #4CAF50; font-weight: bold;'>Target:</span> ${profit_target:.4f} ({profit_target_pct:+.1%})", unsafe_allow_html=True)
            elif asset_type.lower() == 'options':
                # Format options prices as premium
                with entry_cols[0]:
                    st.markdown(f"<span style='color: #2196F3; font-weight: bold;'>Entry Premium:</span> ${entry_price:.2f}", unsafe_allow_html=True)
                with entry_cols[1]:
                    st.markdown(f"<span style='color: #F44336; font-weight: bold;'>Stop-Loss:</span> ${stop_loss:.2f} ({stop_loss_pct:+.1%})", unsafe_allow_html=True)
                with entry_cols[2]:
                    st.markdown(f"<span style='color: #4CAF50; font-weight: bold;'>Target:</span> ${profit_target:.2f} ({profit_target_pct:+.1%})", unsafe_allow_html=True)
            else:
                # Default (stocks and other assets)
                with entry_cols[0]:
                    st.markdown(f"<span style='color: #2196F3; font-weight: bold;'>Entry:</span> ${entry_price:.2f}", unsafe_allow_html=True)
                with entry_cols[1]:
                    st.markdown(f"<span style='color: #F44336; font-weight: bold;'>Stop-Loss:</span> ${stop_loss:.2f} ({stop_loss_pct:+.1%})", unsafe_allow_html=True)
                with entry_cols[2]:
                    st.markdown(f"<span style='color: #4CAF50; font-weight: bold;'>Target:</span> ${profit_target:.2f} ({profit_target_pct:+.1%})", unsafe_allow_html=True)
            
            st.markdown(f"**Exit Conditions:** {exit_conditions}")
            
            # Strategy parameters
            st.markdown("<div class='strategy-params'></div>", unsafe_allow_html=True)
            st.markdown(f"**Strategy Parameters:** {opp.get('strategy_parameters', 'Using default parameters')}")
            
            # Evidence
            st.markdown("<div class='evidence-container'></div>", unsafe_allow_html=True)
            st.markdown(f"**Evidence:** {opp.get('evidence', 'Analysis pending')}")
            
            # Button container inside the card
            st.markdown("<div style='margin-top: 20px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);'></div>", unsafe_allow_html=True)
            
            # More professionally styled buttons (inside the card)
            button_cols = st.columns([1, 1])
            with button_cols[0]:
                opp_id = opp.get('id', 'unknown')
                analysis_button = st.button(f"📊 View Detailed Analysis", key=f"view_{opp_id}", use_container_width=True, type="secondary")
                if analysis_button:
                    st.session_state[f"view_detail_{opp_id}"] = True
            with button_cols[1]:
                # Make the approve button more prominent
                approve_button = st.button(f"✅ Approve for Paper Trading", key=f"approve_{opp_id}", use_container_width=True, type="primary")
                if approve_button:
                    st.session_state[f"approved_{opp_id}"] = True
                    # Add a confirmation animation
                    with st.spinner(f"Approving trade for {opp.get('symbol', f'Opportunity {opp_id}')}..."):
                        # Call the API to approve this opportunity
                        if orchestrator is not None:
                            try:
                                # Approve the opportunity
                                opp_id = opp.get('id', 'unknown')
                                symbol = opp.get('symbol', f'Opportunity {opp_id}')
                                approval_result = orchestrator.approve_opportunity(opp_id)
                                if approval_result.get('status') == 'success':
                                    st.success(f"✅ Successfully approved {symbol} {asset_type_display} trade for paper trading!")
                                    # Add celebration confetti effect
                                    st.balloons()
                                else:
                                    st.error(f"Failed to approve: {approval_result.get('message', 'Unknown error')}")
                            except Exception as e:
                                st.error(f"Error approving opportunity: {e}")
                        else:
                            # Mock success response for demo
                            time.sleep(0.8)  # Simulate API call
                            symbol = opp.get('symbol', f'Opportunity {opp.get("id", "Unknown")}')
                            st.success(f"✅ Successfully approved {symbol} {asset_type_display} trade for paper trading!")
                            # Add celebration confetti effect
                            st.balloons()
            
            # Close the card container div
            st.markdown("</div>", unsafe_allow_html=True)
            
            # If details view is active, show expanded details with enhanced styling
            if st.session_state.get(f"view_detail_{opp.get('id', 'unknown')}", False):
                with st.expander(f"Detailed Analysis for {opp.get('symbol', f"Opportunity {opp.get('id', 'Unknown')}")}", expanded=True):
                    st.markdown("#### Backtest Performance")
                    
                    # Create a 3x2 grid for metrics using native Streamlit components
                    col1, col2, col3 = st.columns(3)
                    
                    # Safely get backtest metrics with defaults
                    backtest = opp.get('backtest', {})
                    if backtest:
                        # Get all metric values with safe defaults
                        return_value = backtest.get('return', 0.0)
                        sharpe_value = backtest.get('sharpe', 0.0)
                        drawdown_value = backtest.get('drawdown', 0.0)
                        win_rate_value = backtest.get('win_rate', 0.0)
                        volatility_value = backtest.get('volatility', 0.0)
                        alpha_value = backtest.get('alpha', 0.0)
                        
                        # Format colors based on values
                        return_color = "#4CAF50" if return_value > 0 else "#F44336"
                        sharpe_color = "#4CAF50" if sharpe_value > 1 else "#FFC107"
                        drawdown_color = "#F44336" if drawdown_value > 0.15 else "#FFC107"
                        winrate_color = "#4CAF50" if win_rate_value > 0.5 else "#FFC107"
                        volatility_color = "#F44336" if volatility_value > 0.2 else "#4CAF50"
                        alpha_color = "#4CAF50" if alpha_value > 0 else "#F44336"
                        
                        # Add metrics with colored values
                        with col1:
                            st.markdown(f"**Total Return**\n<span style='color:{return_color}; font-size:24px; font-weight:bold;'>{return_value:+.2%}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Win Rate**\n<span style='color:{winrate_color}; font-size:24px; font-weight:bold;'>{win_rate_value:.1%}</span>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"**Sharpe Ratio**\n<span style='color:{sharpe_color}; font-size:24px; font-weight:bold;'>{sharpe_value:.2f}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Volatility**\n<span style='color:{volatility_color}; font-size:24px; font-weight:bold;'>{volatility_value:.2%}</span>", unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"**Max Drawdown**\n<span style='color:{drawdown_color}; font-size:24px; font-weight:bold;'>{drawdown_value:.2%}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Benchmark Alpha**\n<span style='color:{alpha_color}; font-size:24px; font-weight:bold;'>{alpha_value:+.2f}</span>", unsafe_allow_html=True)
                    else:
                        st.warning("No backtest data available for this opportunity.")
                
                # Show equity curve if available - using Streamlit's native plotting
                backtest = opp.get('backtest', {})
                if backtest and 'equity_curve' in backtest and PLOTLY_AVAILABLE:
                    
                    # Show equity curve if available
                    if backtest and 'equity_curve' in backtest and 'dates' in backtest and 'benchmark' in backtest and PLOTLY_AVAILABLE:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=backtest['dates'],
                            y=backtest['equity_curve'],
                            mode='lines',
                            name='Strategy',
                            line=dict(color='#2196F3', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=backtest['dates'],
                            y=backtest['benchmark'],
                            mode='lines',
                            name='Benchmark',
                            line=dict(color='#757575', width=1, dash='dash')
                        ))
                        fig.update_layout(
                            title="Equity Curve vs Benchmark",
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value",
                            template="plotly_dark",
                            plot_bgcolor="#1d3b53",
                            paper_bgcolor="#1d3b53",
                            font=dict(color="white"),
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Evidence that led to discovery
                    st.markdown("#### Discovery Evidence")
                    if 'news_items' in opp and opp['news_items']:
                        for news in opp['news_items']:
                            st.markdown(f'''
                            <div style="border-left: 3px solid #2196F3; padding-left: 1rem; margin-bottom: 0.5rem;">
                                <p style="font-weight: bold;">{news['title']}</p>
                                <p>{news['summary']}</p>
                                <p style="opacity: 0.7; font-size: 0.8rem;">{news['source']} | {news['date']} | Sentiment: {news['sentiment']}</p>
                            </div>
                            ''', unsafe_allow_html=True)
                    else:
                        st.markdown("No specific news items associated with this opportunity.")
                    
                    # Technical indicators that triggered the strategy
                    st.markdown("#### Technical Indicators")
                    if 'indicators' in opp and opp['indicators']:
                        indicator_cols = st.columns(3)
                        for i, (name, value) in enumerate(opp['indicators'].items()):
                            with indicator_cols[i % 3]:
                                st.metric(name, f"{value:.2f}")
                    else:
                        st.markdown("No technical indicators associated with this opportunity.")
                    
                    # Close button
                    if st.button("Close Details", key=f"close_{opp['id']}"):
                        st.session_state[f"view_detail_{opp['id']}"] = False
                        st.rerun()

def pipeline_schedule_section():
    """Display and control the autonomous pipeline schedule."""
    # More professional header with clock icon
    st.markdown("<h2>Pipeline Schedule <span style='font-size: 1rem; color: rgba(255,255,255,0.7); font-weight: normal;'>⏱️</span></h2>", unsafe_allow_html=True)
    st.markdown("<p style='margin-bottom: 1.5rem;'>Configure and monitor your automated trading pipeline schedule</p>", unsafe_allow_html=True)
    
    # Show current schedule settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Market Open Schedule")
        market_open_interval = st.number_input(
            "Minutes between runs (market open)", 
            min_value=1, 
            max_value=30, 
            value=st.session_state.get("market_open_interval", 5),
            help="How often to run the pipeline when market is open"
        )
        st.session_state.market_open_interval = market_open_interval
    
    with col2:
        st.markdown("#### Market Closed Schedule")
        market_closed_interval = st.number_input(
            "Minutes between runs (market closed)", 
            min_value=5, 
            max_value=60, 
            value=st.session_state.get("market_closed_interval", 30),
            help="How often to run the pipeline when market is closed"
        )
        st.session_state.market_closed_interval = market_closed_interval
        
        # Asset types to include in pipeline runs
        asset_types = st.multiselect(
            "Asset Classes to Monitor",
            ["Stock", "Crypto", "Forex", "Options", "Futures"],
            default=["Stock", "Crypto", "Forex", "Options"],
            key="schedule_asset_types",
            help="Select which asset classes to scan for opportunities"
        )
        
        # Store selected asset types in session state for pipeline runs
        st.session_state['selected_asset_types'] = [asset.lower() for asset in asset_types]
    
    # Show next scheduled runs
    st.markdown("#### Upcoming Pipeline Runs")
    
    # Get market hours
    now = datetime.now()
    market_open = datetime(now.year, now.month, now.day, 9, 30).replace(tzinfo=now.tzinfo)
    market_close = datetime(now.year, now.month, now.day, 16, 0).replace(tzinfo=now.tzinfo)
    
    # Check if market is currently open
    is_market_open = (
        now.weekday() < 5 and  # Monday to Friday
        market_open <= now <= market_close
    )
    
    # Calculate next few scheduled runs
    scheduled_runs = []
    
    if is_market_open:
        interval_minutes = market_open_interval
        status = "Market Open"
    else:
        interval_minutes = market_closed_interval
        status = "Market Closed"
    
    next_run = now + timedelta(minutes=interval_minutes)
    for i in range(3):  # Show next 3 scheduled runs
        scheduled_runs.append({
            "run_time": next_run.strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "assets": ", ".join(asset_types) if asset_types else "None selected"
        })
        next_run += timedelta(minutes=interval_minutes)
    
    # Display scheduled runs
    runs_df = pd.DataFrame(scheduled_runs)
    st.dataframe(runs_df, use_container_width=True)
    
    # Buttons to control schedule
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Pipeline Now", key="run_now_btn"):
            if orchestrator is not None:
                try:
                    with st.spinner(f"Running pipeline for {', '.join(asset_types)}..."):
                        # Call the orchestrator with filtered asset types
                        result = orchestrator.run_pipeline(asset_types=st.session_state['selected_asset_types'])
                        if result and isinstance(result, list):
                            st.success(f"Pipeline executed successfully! Found {len(result)} opportunities across {len(asset_types)} asset classes.")
                        else:
                            st.warning("Pipeline completed but found no opportunities.")
                except Exception as e:
                    st.error(f"Error running pipeline: {e}")
            else:
                st.warning("Orchestrator not available. Unable to run pipeline.")
    
    with col2:
        if st.button("Save Schedule Settings", key="save_schedule"):
            # Save schedule settings to configuration
            try:
                # Mock saving to config for now
                st.success("Schedule settings saved successfully!")
            except Exception as e:
                st.error(f"Error saving schedule settings: {e}")

# -------------------- MOCK DATA FUNCTIONS FOR AUTONOMOUS TAB --------------------

def get_mock_discoveries():
    """Get mock opportunity discovery data."""
    return [
        {
            "symbol": "AAPL", 
            "timestamp": "2025-04-23 14:32", 
            "source": "News Analysis", 
            "score": 0.87,
            "evidence": "Positive earnings surprise and new product announcement",
            "status": "Testing"
        },
        {
            "symbol": "TSLA", 
            "timestamp": "2025-04-23 15:15", 
            "source": "Market Indicators", 
            "score": 0.76,
            "evidence": "RSI oversold, positive MACD crossover, high volume",
            "status": "Discovered"
        },
        {
            "symbol": "NVDA", 
            "timestamp": "2025-04-23 13:45", 
            "source": "AI Sentiment", 
            "score": 0.92,
            "evidence": "Strong positive sentiment across news, social media",
            "status": "Validating"
        },
        {
            "symbol": "SMCI", 
            "timestamp": "2025-04-23 10:20", 
            "source": "Technical Pattern", 
            "score": 0.81,
            "evidence": "Cup and handle formation, breakout confirmed",
            "status": "Testing"
        },
        {
            "symbol": "META", 
            "timestamp": "2025-04-23 09:45", 
            "source": "News + Technical", 
            "score": 0.79,
            "evidence": "Positive AI news + support level bounce",
            "status": "Discovered"
        }
    ]

def get_mock_backtests():
    """Get mock backtest results data."""
    # Generate date range and curves
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=90)
    dates = [start_date + timedelta(days=i) for i in range(91)]
    dates_str = [d.strftime("%Y-%m-%d") for d in dates]
    
    # AAPL Momentum Strategy
    aapl_equity = [100000]
    for i in range(90):
        change = 100000 * (1 + 0.003 * (np.random.random() - 0.4))
        aapl_equity.append(aapl_equity[-1] + change)
    
    aapl_benchmark = [100000]
    for i in range(90):
        change = 100000 * (1 + 0.001 * (np.random.random() - 0.45))
        aapl_benchmark.append(aapl_benchmark[-1] + change)
    
    # NVDA Mean Reversion Strategy
    nvda_equity = [100000]
    for i in range(90):
        change = 100000 * (1 + 0.005 * (np.random.random() - 0.35))
        nvda_equity.append(nvda_equity[-1] + change)
    
    nvda_benchmark = [100000]
    for i in range(90):
        change = 100000 * (1 + 0.002 * (np.random.random() - 0.45))
        nvda_benchmark.append(nvda_benchmark[-1] + change)
    
    return [
        {
            "symbol": "AAPL",
            "strategy": "Momentum Alpha",
            "return": 0.152,
            "sharpe": 1.85,
            "drawdown": 0.045,
            "win_rate": 0.68,
            "dates": dates_str,
            "equity_curve": aapl_equity,
            "benchmark": aapl_benchmark,
            "optimization_history": [
                {"iteration": 1, "params": {"fast_ma": 8, "slow_ma": 21}, "performance": 0.124},
                {"iteration": 2, "params": {"fast_ma": 10, "slow_ma": 25}, "performance": 0.137},
                {"iteration": 3, "params": {"fast_ma": 12, "slow_ma": 26}, "performance": 0.152}
            ]
        },
        {
            "symbol": "NVDA",
            "strategy": "Mean Reversion Pro",
            "return": 0.213,
            "sharpe": 2.05,
            "drawdown": 0.062,
            "win_rate": 0.72,
            "dates": dates_str,
            "equity_curve": nvda_equity,
            "benchmark": nvda_benchmark,
            "optimization_history": [
                {"iteration": 1, "params": {"rsi_period": 14, "upper_threshold": 70, "lower_threshold": 30}, "performance": 0.162},
                {"iteration": 2, "params": {"rsi_period": 14, "upper_threshold": 75, "lower_threshold": 25}, "performance": 0.185},
                {"iteration": 3, "params": {"rsi_period": 12, "upper_threshold": 75, "lower_threshold": 25}, "performance": 0.201},
                {"iteration": 4, "params": {"rsi_period": 10, "upper_threshold": 80, "lower_threshold": 20}, "performance": 0.213}
            ]
        }
    ]

def get_mock_approved_opportunities():
    """Get mock approved opportunities data."""
    # Generate date range and curves for backtest visualization
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    dates = [start_date + timedelta(days=i) for i in range(31)]
    dates_str = [d.strftime("%Y-%m-%d") for d in dates]
    
    # SMCI Breakout Strategy
    smci_equity = [100000]
    for i in range(30):
        change = 100000 * (1 + 0.008 * (np.random.random() - 0.3))
        smci_equity.append(smci_equity[-1] + change)
    
    smci_benchmark = [100000]
    for i in range(30):
        change = 100000 * (1 + 0.002 * (np.random.random() - 0.45))
        smci_benchmark.append(smci_benchmark[-1] + change)
    
    return [
        {
            "id": "opp_001",
            "expected_return": 0.182,
            "sharpe": 2.15,
            "confidence": 0.89,
            "entry_price": 788.45,
            "stop_loss": 742.50,
            "profit_target": 880.00,
            "position_size": 12500,
            "exit_conditions": "Exit if RSI > 78 or ATR contracts by more than 25% from entry",
            "strategy_parameters": "Using Price Channel(20) breakout, Volume filter: 2.0x 10-day average, ATR-based position sizing, Trailing stop: 2.5x daily ATR",
            "evidence": "Strong technical breakout with high volume and positive sector momentum",
            "submission_time": (datetime.now() - timedelta(hours=4, minutes=32)).strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Ready for Approval",
            "discovery_reason": "High sentiment and volatility",
            "backtest": {
                "return": 0.341,
                "sharpe": 2.15,
                "drawdown": 0.126,
                "win_rate": 0.68,
                "volatility": 0.17,
                "alpha": 0.12,
                "dates": pd.date_range(start=datetime.now() - timedelta(days=180), periods=180, freq='D'),
                "equity_curve": np.cumsum(np.random.normal(0.002, 0.015, 180)),
                "benchmark": np.cumsum(np.random.normal(0.001, 0.012, 180)),
                "optimization_history": [
                    {"iteration": 1, "parameters": {"period": 15, "vol_filter": 1.5}, "performance": 0.28},
                    {"iteration": 2, "parameters": {"period": 20, "vol_filter": 2.0}, "performance": 0.341}
                ]
            },
            "indicators": {
                "RSI": 58.7,
                "MACD": 1.45,
                "Volume Ratio": 2.38,
                "ATR": 3.25,
                "Bollinger %B": 0.65
            }
        },
        {
            "id": "opp_003",
            "symbol": "NVDA",
            "strategy": "Momentum Alpha",
            "expected_return": 0.156,
            "sharpe": 1.93,
            "confidence": 0.86,
            "submission_time": (datetime.now() - timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Ready for Approval",
            "evidence": "Consistent record high volume days with positive price action and sector strength",
            "asset_type": "Stock",
            "position_size": 15000,
            "entry_price": 925.70,
            "stop_loss": 875.50,
            "profit_target": 1025.00,
            "exit_conditions": "Exit if EMA(5) crosses below EMA(20) or after 20 trading days",
            "strategy_parameters": "Using EMA(9,21) crossover, Money Flow Index > 70, Relative strength vs. sector > 1.2, EMA slope filter",
            "backtest": {
                "return": 0.156,
                "sharpe": 1.93,
                "drawdown": 0.047,
                "win_rate": 0.71,
                "volatility": 0.028,
                "alpha": 0.85,
                "dates": dates_str,
                "equity_curve": [v * 1.05 for v in smci_equity],  # Reuse with adjustment
                "benchmark": [v * 0.98 for v in smci_benchmark]  # Reuse with adjustment
            },
            "news_items": [
                {
                    "title": "NVIDIA Expands AI Cloud Infrastructure Partnerships",
                    "summary": "NVIDIA announced expanded partnerships with major cloud providers to deploy its latest H200 GPUs for AI training and inference workloads.",
                    "source": "Tech Insider",
                    "date": "2025-04-22",
                    "sentiment": "Positive"
                }
            ],
            "indicators": {
                "RSI": 67.8,
                "MACD": 3.56,
                "Volume Ratio": 2.85,
                "ATR": 12.35,
                "Bollinger %B": 0.78
            }
        }
    ]

# -------------------- AUTONOMOUS TAB COMPONENTS --------------------

def regime_badge(regime, confidence):
    """Display market regime with confidence score.
    
    Args:
        regime (str): Current market regime (e.g., "Bullish Trend")
        confidence (float): Confidence level (0-1)
    """
    # Map regimes to colors
    regime_colors = {
        "bullish": "#4caf50",  # Green
        "bearish": "#f44336",  # Red
        "neutral": "#ff9800",  # Amber
        "volatility": "#9c27b0",  # Purple
    }
    
    # Determine color based on regime name
    color = "#4caf50"  # Default green
    for key, value in regime_colors.items():
        if key in regime.lower():
            color = value
            break
    
    st.markdown(f'''
    <div class="card">
        <div class="card-header">Market Regime</div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 1.5rem; font-weight: bold; color: {color};">
                {regime.title()}
            </div>
            <div>
                <div class="text-muted" style="font-size: 0.9rem;">Confidence</div>
                <div style="font-size: 1.2rem; font-weight: bold;">{confidence:.0%}</div>
            </div>
        </div>
        <div style="font-size: 0.85rem; margin-top: 0.8rem; opacity: 0.7;">
            <span>Active since 14 days • </span>
            <span>Last change: {(datetime.now() - timedelta(days=14)).strftime('%b %d, %Y')}</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def performance_metrics(total_return, sharpe, max_drawdown, win_rate):
    """Display key performance metrics of the autonomous system.
    
    Args:
        total_return (float): Total return percentage (e.g., 0.158 for 15.8%)
        sharpe (float): Sharpe ratio (e.g., 1.94)
        max_drawdown (float): Maximum drawdown (e.g., 0.085 for 8.5%)
        win_rate (float): Win rate (e.g., 0.72 for 72%)
    """
    # Format the metrics
    total_return_str = f"{total_return:+.1%}"
    total_return_color = "#4caf50" if total_return > 0 else "#f44336"
    
    sharpe_str = f"{sharpe:.2f}"
    sharpe_color = "#4caf50" if sharpe > 1.0 else ("#ff9800" if sharpe > 0 else "#f44336")
    
    max_drawdown_str = f"{max_drawdown:.1%}"
    max_drawdown_color = "#f44336"  # Always red for drawdown
    
    win_rate_str = f"{win_rate:.0%}"
    win_rate_color = "#4caf50" if win_rate > 0.5 else ("#ff9800" if win_rate > 0.4 else "#f44336")
    
    st.markdown(f'''
    <div class="card">
        <div class="card-header">Bot Performance Summary</div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
            <div>
                <div class="text-muted" style="font-size: 0.9rem;">Total Return</div>
                <div style="font-size: 1.3rem; font-weight: bold; color: {total_return_color};">{total_return_str}</div>
            </div>
            <div>
                <div class="text-muted" style="font-size: 0.9rem;">Sharpe Ratio</div>
                <div style="font-size: 1.3rem; font-weight: bold; color: {sharpe_color};">{sharpe_str}</div>
            </div>
            <div>
                <div class="text-muted" style="font-size: 0.9rem;">Max Drawdown</div>
                <div style="font-size: 1.3rem; font-weight: bold; color: {max_drawdown_color};">{max_drawdown_str}</div>
            </div>
            <div>
                <div class="text-muted" style="font-size: 0.9rem;">Win Rate</div>
                <div style="font-size: 1.3rem; font-weight: bold; color: {win_rate_color};">{win_rate_str}</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def strategy_cards(strategies):
    """Display strategy cards with performance metrics.
    
    Args:
        strategies (list): List of strategy dictionaries with name, type, status, performance, and sharpe
    """
    st.markdown('<div class="card"><div class="card-header">Active Strategies</div>', unsafe_allow_html=True)
    
    # Create columns based on number of strategies (up to 3 per row)
    cols = st.columns(min(len(strategies), 3))
    
    # Display each strategy in its own column
    for i, strategy in enumerate(strategies):
        status_color = "#4CAF50" if strategy["status"] == "Active" else "#FF9800"
        perf_color = "#4CAF50" if strategy["performance"] > 0 else "#F44336"
        
        with cols[i % 3]:
            # Use individual components instead of one big HTML block
            st.markdown(f'''
            <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="font-weight: bold; font-size: 1.1rem;">{strategy["name"]}</div>
                    <div style="background-color: {status_color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8rem;">{strategy["status"]}</div>
                </div>
                <div style="color: rgba(255,255,255,0.7); margin-bottom: 15px; font-size: 0.9rem;">{strategy["type"]}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Performance and Sharpe metrics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'''
                <div style="margin-bottom: 15px;">
                    <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Performance</div>
                    <div style="font-size: 1.4rem; font-weight: bold; color: {perf_color};">{strategy["performance"]:+.1%}</div>
                </div>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown(f'''
                <div style="margin-bottom: 15px;">
                    <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Sharpe</div>
                    <div style="font-size: 1.4rem; font-weight: bold;">{strategy["sharpe"]:.2f}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            # Add session state keys for this strategy if not present
            if f"show_details_{i}" not in st.session_state:
                st.session_state[f"show_details_{i}"] = False
                
            # Action buttons using Streamlit native buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Details", key=f"details_{i}", use_container_width=True):
                    st.session_state[f"show_details_{i}"] = True
            with col2:
                st.button("Pause", key=f"pause_{i}", use_container_width=True, type="secondary")
                
            # Show details modal if button was clicked
            if st.session_state.get(f"show_details_{i}", False):
                with st.expander(f"**{strategy['name']} Details**", expanded=True):
                    st.markdown(f"### {strategy['name']} Strategy Overview")
                    
                    # Strategy description based on type
                    if "momentum" in strategy["type"].lower():
                        st.markdown("""
                        **What is Momentum Trading?**
                        
                        Momentum trading is a strategy that aims to capitalize on the continuance of existing market trends. 
                        It's based on the idea that assets that have performed well will continue to perform well in the short term.
                        
                        **Key Principles:**
                        * Buys securities that are rising in price
                        * Sells securities that are falling in price
                        * Uses technical indicators like moving averages and relative strength
                        * Typically shorter-term focused (days to weeks)
                        
                        **Best Market Conditions:** Strong trending markets with clear direction
                        
                        **Risks:** Sudden market reversals, increased volatility
                        """)
                    elif "reversion" in strategy["type"].lower():
                        st.markdown("""
                        **What is Mean Reversion Trading?**
                        
                        Mean reversion trading is based on the theory that prices and returns eventually move back toward their historical average or mean.
                        This strategy looks for extreme price movements and bets on a return to normal levels.
                        
                        **Key Principles:**
                        * Buys securities that have fallen significantly below their average
                        * Sells securities that have risen significantly above their average
                        * Uses indicators like RSI, Bollinger Bands, and standard deviation
                        * Can be short to medium-term focused
                        
                        **Best Market Conditions:** Range-bound or oscillating markets
                        
                        **Risks:** Extended trends that don't revert, fundamental shifts in value
                        """)
                    elif "sentiment" in strategy["type"].lower() or "nlp" in strategy["type"].lower():
                        st.markdown("""
                        **What is AI Sentiment Trading?**
                        
                        AI Sentiment trading uses natural language processing and machine learning to analyze news, social media, and other text sources
                        to gauge market sentiment about specific securities or the market as a whole.
                        
                        **Key Principles:**
                        * Analyzes large volumes of text data from multiple sources
                        * Identifies bullish or bearish sentiment signals
                        * Combines sentiment with traditional indicators for signal confirmation
                        * Can be applied to various timeframes
                        
                        **Best Market Conditions:** News-driven markets, high information flow
                        
                        **Risks:** Sentiment misinterpretation, overwhelming noise, data source limitations
                        """)
                    else:
                        st.markdown(f"**{strategy['type']}** is a sophisticated trading strategy that combines multiple market signals to identify potential trading opportunities.")
                        
                    # Performance metrics with explanations
                    st.markdown("### Current Performance")
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        st.metric("Return", f"{strategy['performance']:+.1%}")
                        st.markdown("*How much money this strategy has made since activation*")
                        
                    with metrics_col2:
                        st.metric("Sharpe Ratio", f"{strategy['sharpe']:.2f}")
                        st.markdown("*Risk-adjusted performance (higher is better, >1 is good)*")
                        
                    # Add close button
                    if st.button("Close", key=f"close_details_{i}"):
                        st.session_state[f"show_details_{i}"] = False
                        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def critical_alerts(alerts):
    """Display critical system alerts.
    
    Args:
        alerts (list): List of alert dictionaries with id, type, message, and timestamp
    """
    st.markdown('<div class="card"><div class="card-header">Critical Alerts</div>', unsafe_allow_html=True)
    
    if not alerts:
        st.markdown('<div style="text-align: center; padding: 2rem; opacity: 0.7;">No alerts at this time</div>', unsafe_allow_html=True)
    else:
        for alert in alerts:
            # Map alert types to icons and colors
            icon = "⚠️"
            if alert["type"] == "bullish":
                icon = "📈"
                badge_class = "badge-success"
            elif alert["type"] == "bearish":
                icon = "📉"
                badge_class = "badge-danger"
            elif alert["type"] == "trend":
                icon = "📊"
                badge_class = "badge-warning"
            elif alert["type"] == "volatility":
                icon = "📊"
                badge_class = "badge-warning"
            else:
                badge_class = "badge-warning"
            
            st.markdown(f'''
            <div style="padding: 0.7rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <div>
                        <span class="badge {badge_class}" style="margin-right: 0.5rem;">{alert["type"].title()}</span>
                        <strong>{icon} {alert["message"]}</strong>
                    </div>
                    <div class="text-muted" style="font-size: 0.9rem;">{alert["timestamp"]}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def pipeline_control(last_run, status, on_rerun):
    """Display pipeline control panel with last run info and rerun button.
    
    Args:
        last_run (str): ISO timestamp of last pipeline run
        status (str): Current pipeline status ('ok', 'error', 'running')
        on_rerun (callable): Callback function for rerun button
    """
    # Format status indicator
    if status == "ok":
        status_html = '<span class="status-dot status-active"></span> Pipeline Ready'
    elif status == "error":
        status_html = '<span class="status-dot status-inactive"></span> Pipeline Error'
    elif status == "running":
        status_html = '<span class="status-dot status-warning"></span> Pipeline Running'
    else:
        status_html = '<span class="status-dot"></span> Unknown Status'
    
    # Format last run time
    if last_run:
        last_run_formatted = last_run
    else:
        last_run_formatted = "Never"
    
    st.markdown(f'''
    <div class="card">
        <div class="card-header">Pipeline Control</div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="margin-bottom: 0.5rem;">{status_html}</div>
                <div class="text-muted">Last Run: {last_run_formatted}</div>
            </div>
            <div>
                <button id="rerun-pipeline" class="button button-primary">⟳ Rerun Pipeline</button>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Handle button click (Streamlit doesn't support JavaScript button clicks, so we use Streamlit buttons)
    if st.button("⟳ Rerun Pipeline", key="rerun_pipeline_btn"):
        on_rerun()

# -------------------- MOCK DATA FUNCTIONS --------------------

def get_regime_data():
    """Get real market regime data from the orchestrator or data hub if available, otherwise use mock data."""
    # Try to get real regime data from orchestrator
    if orchestrator is not None:
        try:
            # Get the latest regime assessment from the orchestrator
            regime_assessment = orchestrator.get_current_market_regime()
            if regime_assessment:
                # Extract regime name and confidence
                regime_name = regime_assessment.get("regime", "Neutral")
                confidence = regime_assessment.get("confidence", 0.5)
                last_change = regime_assessment.get("last_change_date")
                
                return {
                    "regime": regime_name,
                    "confidence": confidence,
                    "last_change": last_change,
                    "is_real_data": True
                }
        except Exception as e:
            print(f"Error getting real regime data: {e}")
    
    # Try to get regime from data hub as fallback
    if data_hub is not None:
        try:
            market_indicators = data_hub.get_market_indicators(["vix", "macd", "rsi"])
            
            # Simplified regime determination based on indicators
            vix = market_indicators.get("vix", {}).get("value", 20)
            rsi = market_indicators.get("rsi", {}).get("value", 50)
            macd_signal = market_indicators.get("macd", {}).get("signal", 0)
            
            # Determine regime based on indicators
            if rsi > 70 and macd_signal > 0:  # Overbought and positive trend
                regime = "Bullish Extended"
                confidence = 0.85
            elif rsi < 30 and macd_signal < 0:  # Oversold and negative trend 
                regime = "Bearish Extended"
                confidence = 0.85
            elif rsi > 60 and macd_signal > 0:  # Healthy uptrend
                regime = "Bullish Trend"
                confidence = 0.75
            elif rsi < 40 and macd_signal < 0:  # Healthy downtrend
                regime = "Bearish Trend"
                confidence = 0.75
            elif vix > 30:  # High volatility
                regime = "Volatility Regime"
                confidence = 0.80
            elif 45 < rsi < 55:  # Neutral
                regime = "Neutral Consolidation"
                confidence = 0.65
            else:  # Default
                regime = "Mixed Signals"
                confidence = 0.60
                
            return {
                "regime": regime,
                "confidence": confidence,
                "last_change": datetime.now() - timedelta(days=7),
                "is_real_data": True
            }
        except Exception as e:
            print(f"Error getting regime from indicators: {e}")
    
    # Fallback to mock data
    return {
        "regime": "Bullish Neutral",
        "confidence": 0.91,
        "last_change": datetime.now() - timedelta(days=14),
        "is_real_data": False
    }

def get_performance_data():
    """Get real performance metrics from the orchestrator or data hub if available, otherwise use mock data."""
    # Try to get real performance data from orchestrator
    if orchestrator is not None:
        try:
            # Get the latest performance summary from the orchestrator
            performance = orchestrator.get_performance_summary()
            if performance:
                return {
                    "total_return": performance.get("total_return", 0.0),
                    "sharpe_ratio": performance.get("sharpe_ratio", 0.0),
                    "max_drawdown": performance.get("max_drawdown", 0.0),
                    "win_rate": performance.get("win_rate", 0.0),
                    "is_real_data": True
                }
        except Exception as e:
            print(f"Error getting real performance data: {e}")
    
    # Try to get performance from data hub as fallback
    if data_hub is not None:
        try:
            # Get performance metrics from data hub
            equity_curve = data_hub.get_performance_history(lookback_days=90)
            if equity_curve is not None and len(equity_curve) > 0:
                # Calculate metrics from equity curve
                start_value = equity_curve[0]["value"] if len(equity_curve) > 0 else 100000
                end_value = equity_curve[-1]["value"] if len(equity_curve) > 0 else 100000
                total_return = (end_value - start_value) / start_value
                
                # Calculate other metrics
                daily_returns = []
                peak = start_value
                max_drawdown = 0
                wins = 0
                trades = 0
                
                for i in range(1, len(equity_curve)):
                    # Daily return
                    daily_return = (equity_curve[i]["value"] - equity_curve[i-1]["value"]) / equity_curve[i-1]["value"]
                    daily_returns.append(daily_return)
                    
                    # Update peak and drawdown
                    peak = max(peak, equity_curve[i]["value"])
                    drawdown = (peak - equity_curve[i]["value"]) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                    
                    # Count trades and wins if available
                    if "trades" in equity_curve[i]:
                        trades += equity_curve[i]["trades"]
                        if "wins" in equity_curve[i]:
                            wins += equity_curve[i]["wins"]
                
                # Calculate Sharpe ratio
                if len(daily_returns) > 0:
                    mean_return = sum(daily_returns) / len(daily_returns)
                    std_return = (sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5
                    risk_free_rate = 0.02 / 252  # Daily risk-free rate (2% annual)
                    sharpe = (mean_return - risk_free_rate) / std_return * (252 ** 0.5) if std_return > 0 else 0
                else:
                    sharpe = 0
                
                # Calculate win rate
                win_rate = wins / trades if trades > 0 else 0.5
                
                return {
                    "total_return": total_return,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate,
                    "is_real_data": True
                }
        except Exception as e:
            print(f"Error calculating performance from data hub: {e}")
    
    # Fallback to mock data
    return {
        "total_return": 0.158,
        "sharpe_ratio": 1.94,
        "max_drawdown": 0.085,
        "win_rate": 0.72,
        "is_real_data": False
    }

def get_strategies():
    """Get real strategy data from the orchestrator if available, otherwise use mock data."""
    # Try to get real strategy data from orchestrator
    if orchestrator is not None:
        try:
            # Get the active strategies from the orchestrator
            active_strategies = orchestrator.get_active_strategies()
            if active_strategies and len(active_strategies) > 0:
                # Transform to expected format
                strategies = []
                for strategy in active_strategies:
                    status = "Active"
                    if "status" in strategy:
                        status = strategy["status"]
                    elif "is_active" in strategy:
                        status = "Active" if strategy["is_active"] else "Inactive"
                    
                    strategies.append({
                        "name": strategy.get("name", "Unknown Strategy"),
                        "type": strategy.get("type", "Custom"),
                        "status": status,
                        "performance": strategy.get("performance", 0.0),
                        "sharpe": strategy.get("sharpe_ratio", 0.0),
                        "is_real_data": True
                    })
                return strategies
        except Exception as e:
            print(f"Error getting real strategy data: {e}")
    
    # Try to get strategy data from data hub as fallback
    if data_hub is not None:
        try:
            # Get strategy allocation from data hub
            strategy_allocation = data_hub.get_strategy_allocation()
            if strategy_allocation and len(strategy_allocation) > 0:
                # Get strategy performance
                strategy_performance = data_hub.get_strategy_performance()
                
                # Combine allocation with performance data
                strategies = []
                for name, allocation in strategy_allocation.items():
                    # Get performance for this strategy if available
                    perf = 0.0
                    sharpe = 0.0
                    if strategy_performance and name in strategy_performance:
                        perf = strategy_performance[name].get("return", 0.0)
                        sharpe = strategy_performance[name].get("sharpe", 0.0)
                    
                    # Determine status based on allocation
                    status = "Inactive"
                    if allocation > 0.05:  # More than 5% allocation
                        status = "Active"
                    elif 0 < allocation <= 0.05:  # Less than 5% allocation
                        status = "Testing"
                    
                    # Determine type based on name
                    strategy_type = "Custom"
                    if "momentum" in name.lower():
                        strategy_type = "Momentum"
                    elif "reversion" in name.lower() or "mean" in name.lower():
                        strategy_type = "Mean Reversion"
                    elif "sentiment" in name.lower() or "nlp" in name.lower() or "ai" in name.lower():
                        strategy_type = "NLP/Sentiment"
                    elif "breakout" in name.lower():
                        strategy_type = "Breakout"
                    elif "trend" in name.lower() or "follow" in name.lower():
                        strategy_type = "Trend Following"
                    
                    strategies.append({
                        "name": name,
                        "type": strategy_type,
                        "status": status,
                        "performance": perf,
                        "sharpe": sharpe,
                        "allocation": allocation,
                        "is_real_data": True
                    })
                
                return strategies
        except Exception as e:
            print(f"Error getting strategy data from data hub: {e}")
    
    # Fallback to mock data
    return [
        {
            "name": "Momentum Alpha",
            "type": "Momentum",
            "status": "Active",
            "performance": 0.152,
            "sharpe": 1.85,
            "is_real_data": False
        },
        {
            "name": "Mean Reversion Pro",
            "type": "Mean Reversion",
            "status": "Active",
            "performance": 0.087,
            "sharpe": 1.42,
            "is_real_data": False
        },
        {
            "name": "AI Sentiment Edge",
            "type": "NLP/Sentiment",
            "status": "Testing",
            "performance": 0.213,
            "sharpe": 2.05,
            "is_real_data": False
        }
    ]

def get_alerts():
    """Get real alert data from AlertManager if available, fallback to mock data."""
    # Try to get real alerts from alert manager
    if alert_manager is not None:
        try:
            # Get the top 5 alerts from alert manager
            raw_alerts = alert_manager.get_recent_alerts(limit=5)
            
            # Transform alerts to expected format
            alerts = []
            for alert in raw_alerts:
                alert_type = "trend"
                if "bull" in alert.category.lower():
                    alert_type = "bullish"
                elif "bear" in alert.category.lower():
                    alert_type = "bearish"
                elif "volatility" in alert.category.lower():
                    alert_type = "volatility"
                
                alerts.append({
                    "id": alert.id,
                    "type": alert_type,
                    "message": alert.message,
                    "timestamp": alert.timestamp.strftime("%Y-%m-%d %H:%M")
                })
            
            return alerts
        except Exception as e:
            print(f"Error getting real alerts: {e}")

    # Use newsService as a secondary source for alerts
    if news_service is not None:
        try:
            # Get high impact news as alerts
            news_items = news_service.get_market_news(impact_filter="high", limit=3)
            alerts = []
            
            for idx, news in enumerate(news_items):
                sentiment = news.get("sentiment", "neutral")
                alert_type = "trend"
                
                if sentiment == "positive":
                    alert_type = "bullish"
                elif sentiment == "negative":
                    alert_type = "bearish"
                
                alerts.append({
                    "id": f"news_{idx}",
                    "type": alert_type,
                    "message": news.get("title", "Market news alert"),
                    "timestamp": news.get("published_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
                })
            
            return alerts
        except Exception as e:
            print(f"Error getting news as alerts: {e}")

    # Fallback to mock data
    return [
        {
            "id": "alert001",
            "type": "trend",
            "message": "Market regime shifted from Bullish to Bullish Neutral",
            "timestamp": "2025-04-23 18:30"
        },
        {
            "id": "alert002",
            "type": "volatility",
            "message": "VIX increased by 15% in the last hour",
            "timestamp": "2025-04-23 15:45"
        },
        {
            "id": "alert003",
            "type": "bullish",
            "message": "Increased allocation to Mean Reversion strategy",
            "timestamp": "2025-04-23 14:20"
        }
    ]

def rerun_pipeline():
    """Rerun the autonomous pipeline using the orchestrator if available."""
    pipeline_status = "error"
    
    st.session_state.last_pipeline_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get asset types from session state, default to all asset types
    asset_types = st.session_state.get('selected_asset_types', ['stock', 'crypto', 'forex', 'options', 'futures'])
    
    # Try to run the real pipeline
    if orchestrator is not None:
        try:
            with st.spinner(f"Running autonomous pipeline..."):
                # Set pipeline status to running
                st.session_state.pipeline_status = "running"
                
                # Call the orchestrator's pipeline with asset types filter
                pipeline_result = orchestrator.run_pipeline(asset_types=asset_types)
                
                # Handle the result
                if pipeline_result and isinstance(pipeline_result, list):
                    st.session_state.pipeline_status = "ok"
                    return "ok"
                else:
                    st.session_state.pipeline_status = "error"
                    return "error"
        except Exception as e:
            st.error(f"Pipeline execution failed: {e}")
            st.session_state.pipeline_status = "error"
            return "error"
    
    # If no orchestrator, simulate a successful run
    st.session_state.pipeline_status = "ok"
    return "ok"

# -------------------- DASHBOARD TAB --------------------
def render_dashboard_tab():
    """Render the Dashboard tab with all components."""
    # Header with symbol selector
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown('''
        <div class="dashboard-header">
            <h1>🤖 Autonomous Trading Dashboard</h1>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        # Symbol selector in the header
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN"]
        selected_symbol = st.selectbox("Market Focus", symbols, index=0, key="market_focus")
        st.session_state.active_symbol = selected_symbol
    
    # Subtitle
    st.markdown("_Mission Control: key metrics & status of your autonomous bot_")
    st.markdown("")
    
    # Initialize session state for pipeline status if not present
    if "pipeline_status" not in st.session_state:
        st.session_state.pipeline_status = "ok"
    
    # Get data using real services when available with fallback to mock data
    with st.spinner("Loading dashboard data..."):
        try:
            # Get market regime data
            regime_data = get_regime_data()
            
            # Get performance metrics
            performance_data = get_performance_data()
            
            # Get active strategies
            strategies_data = get_strategies()
            
            # Get alerts
            alerts_data = get_alerts()
            
            # Check if we're using real data and show indicator
            if any([d.get("is_real_data", False) for d in [regime_data, performance_data]]) or \
               any([s.get("is_real_data", False) for s in strategies_data]):
                st.success("✅ Dashboard is using real data from your autonomous trading system")
            else:
                st.info("ℹ️ Using simulation data - connect your autonomous system for real data")
                
        except Exception as e:
            st.error(f"Error loading dashboard data: {e}")
            # Fallback to mock data in case of errors
            if "regime_data" not in locals() or regime_data is None:
                regime_data = get_regime_data()
            if "performance_data" not in locals() or performance_data is None:
                performance_data = get_performance_data()
            if "strategies_data" not in locals() or strategies_data is None:
                strategies_data = get_strategies()
            if "alerts_data" not in locals() or alerts_data is None:
                alerts_data = get_alerts()
    
    # Top row: Regime badge and performance metrics
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Market regime badge
        regime_badge(regime_data["regime"], regime_data["confidence"])
    
    with col2:
        # Performance metrics
        performance_metrics(
            performance_data["total_return"],
            performance_data["sharpe_ratio"],
            performance_data["max_drawdown"],
            performance_data["win_rate"]
        )
    
    # Middle row: Strategy allocation cards
    strategy_cards(strategies_data)
    
    # Bottom row: Critical alerts and pipeline control
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Critical alerts
        critical_alerts(alerts_data)
    
    with col2:
        # Pipeline control
        pipeline_control(
            st.session_state.get("last_pipeline_run", "Never"),
            st.session_state.get("pipeline_status", "ok"),  # Use pipeline status from session state
            rerun_pipeline
        )

# -------------------- AUTONOMOUS TAB --------------------
def render_autonomous_tab():
    """Render the Autonomous tab showing opportunity discovery, testing, and approval workflow."""
    # Header
    st.markdown("""
    <div style="padding-bottom: 1rem;">
        <h1>Autonomous Trading Pipeline</h1>
        <p>Review and approve autonomous trading opportunities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add asset type filter options
    st.markdown("### Filter Opportunities")
    
    # Create asset type filter with multiselect
    col1, col2 = st.columns([3, 1])
    
    with col1:
        asset_filter = st.multiselect(
            "Asset Types",
            options=["Stock", "Crypto", "Forex", "Options", "Futures"],
            default=["Stock", "Crypto", "Forex", "Options"],
            help="Select which asset classes to display opportunities for"
        )
        
        # Convert display names back to lowercase for filtering
        asset_filter_lowercase = [asset.lower() for asset in asset_filter]
        
        # Store selected asset types in session state for pipeline runs
        if asset_filter:
            st.session_state['selected_asset_types'] = [asset.lower() for asset in asset_filter]
    
    with col2:
        # Add a button to refresh/rerun pipeline with selected asset types
        if st.button("🔄 Refresh Pipeline", use_container_width=True):
            with st.spinner("Running autonomous pipeline for selected asset types..."):
                if orchestrator is not None:
                    try:
                        # Convert selected asset types to lowercase for API
                        asset_types = [asset.lower() for asset in asset_filter]
                        # Run pipeline with asset type filtering
                        pipeline_result = orchestrator.run_pipeline(asset_types=asset_types)
                        if pipeline_result:
                            st.success(f"Found {len(pipeline_result)} opportunities across {len(asset_filter)} asset classes")
                        else:
                            st.warning("No opportunities found for the selected asset types")
                    except Exception as e:
                        st.error(f"Error running pipeline: {e}")
                else:
                    # Mock success for demo
                    time.sleep(0.8)
                    st.success(f"Pipeline refreshed with {len(asset_filter)} asset types")
    
    st.markdown("<hr style='margin: 1.5rem 0'>", unsafe_allow_html=True)
    
    # Get approved opportunities
    all_opportunities = get_mock_approved_opportunities()
    
    # Filter opportunities by selected asset types
    if asset_filter:
        opportunities = [opp for opp in all_opportunities if opp.get('asset_type', 'stock').lower() in asset_filter_lowercase]
    else:
        opportunities = all_opportunities  # Show all if nothing selected
    
    # Show the approval queue at the top (most important for quick decisions)
    approval_queue_section(opportunities)
    
    # Show opportunity discovery section
    opportunity_discovery_section(get_mock_discoveries())
    
    # Show backtest results section
    backtest_results_section(get_mock_backtests())
    
    # Show pipeline schedule controls
    pipeline_schedule_section()

def pipeline_schedule_section():
    """Display and control the autonomous pipeline schedule."""
    # Pipeline scheduling section
    st.markdown("<h2>Pipeline Schedule <span style='font-size: 1rem; color: rgba(255,255,255,0.7); font-weight: normal;'>⏱️</span></h2>", unsafe_allow_html=True)
    st.markdown("<p style='margin-bottom: 1.5rem;'>Configure and monitor your automated trading pipeline schedule</p>", unsafe_allow_html=True)
    
    # Initialize variables for safety
    discoveries = []
    backtests = []
    approved_opportunities = []
    
    # Try to get approved opportunities
    if orchestrator is not None:
        discoveries = orchestrator.discover_symbols() 
        backtests = []
    else:
        # Use mock data
        discoveries = get_mock_discoveries()
        backtests = get_mock_backtests()
        approved_opportunities = get_mock_approved_opportunities()
    
    # Show data source indicator
    if orchestrator is not None:
        st.success("✅ Connected to your autonomous trading pipeline")
    else:
        st.warning("⚠️ Using MOCK DATA - connect to your trading system for live data")

    # Show the approval queue at the top (most important for quick decisions)
    approval_queue_section(approved_opportunities)
    
    # Show opportunity discovery section
    opportunity_discovery_section(discoveries)
    
    # Show backtest results section
    backtest_results_section(backtests)
    
    # Show pipeline schedule controls
    pipeline_schedule_section()

# -------------------- MAIN APP LAYOUT --------------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Dashboard"
    
# Initialize chat widget related session state variables
if 'chat_shown' not in st.session_state:
    st.session_state.chat_shown = False
    
if 'chat_widget_position' not in st.session_state:
    st.session_state.chat_widget_position = {
        'bottom': '80px',
        'right': '20px',
        'display': 'flex'
    }

# Tab navigation
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Dashboard", use_container_width=True, type="primary" if st.session_state.active_tab == "Dashboard" else "secondary"):
        st.session_state.active_tab = "Dashboard"
        # Don't reset chat_shown state when changing tabs to maintain persistence
        st.rerun()
with col2:
    if st.button("Autonomous", use_container_width=True, type="primary" if st.session_state.active_tab == "Autonomous" else "secondary"):
        st.session_state.active_tab = "Autonomous"
        # Don't reset chat_shown state when changing tabs to maintain persistence
        st.rerun()
with col3:
    if st.button("News", use_container_width=True, type="primary" if st.session_state.active_tab == "News" else "secondary"):
        st.session_state.active_tab = "News"
        # Don't reset chat_shown state when changing tabs to maintain persistence
        st.rerun()
with col4:
    if st.button("Settings", use_container_width=True, type="primary" if st.session_state.active_tab == "Settings" else "secondary"):
        st.session_state.active_tab = "Settings"
        # Don't reset chat_shown state when changing tabs to maintain persistence
        st.rerun()

# Divider
st.markdown("<hr>", unsafe_allow_html=True)

# Render the appropriate tab based on selection
if st.session_state.active_tab == "Dashboard":
    render_dashboard_tab()
elif st.session_state.active_tab == "Autonomous":
    render_autonomous_tab()
elif st.session_state.active_tab == "News":
    st.info("News & Analysis tab will be implemented next")
elif st.session_state.active_tab == "Settings":
    st.info("Settings tab will be implemented after that")
else:
    render_dashboard_tab()

# In the sidebar, add a simple button that definitely works
with st.sidebar:
    st.markdown("### AI Trading Assistant")
    if st.button("Open AI Chat", key="reliable_chat_btn", use_container_width=True, type="primary"):
        # Enable chat and force a rerun
        st.session_state.chat_shown = True
        st.rerun()

# Add a much simpler direct button approach instead of fancy styling
# This ensures maximum compatibility across all tabs
st.markdown("""
<style>
    /* Simple green button fixed at bottom right */
    #simple-chat-btn {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 50%;
        font-size: 24px;
        cursor: pointer;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        z-index: 9999;
    }
    
    #simple-chat-btn:hover {
        background-color: #45a049;
        box-shadow: 0 6px 10px rgba(0,0,0,0.4);
    }
</style>

<button id="simple-chat-btn" onclick="directOpenChat()" title="Open AI Trading Assistant">💬</button>

<script>
    // Direct function that manipulates session state
    function directOpenChat() {
        const currentTab = window.location.hash || '#Dashboard';
        console.log('Opening chat on tab: ' + currentTab);
        
        // Create a form to post data directly to Streamlit
        const form = document.createElement('form');
        form.method = 'POST';
        form.style.display = 'none';
        
        // Add hidden input for session state modification
        const input = document.createElement('input');
        input.type = 'hidden';
        input.name = 'chat_shown';
        input.value = 'true';
        form.appendChild(input);
        
        // Append to document and submit
        document.body.appendChild(form);
        form.submit();
        
        // Fallback - also try clicking any "Open AI Assistant" button
        const buttons = document.querySelectorAll('button');
        for (let btn of buttons) {
            if (btn.innerText.includes('Open AI Assistant') || 
                btn.innerText.includes('Open Chat Assistant')) {
                btn.click();
                console.log('Found and clicked assistant button');
                break;
            }
        }
        
        // Ultimate fallback - reload with special query param
        if (window.location.search.indexOf('show_chat=true') === -1) {
            const separator = window.location.search ? '&' : '?';
            window.location.href = window.location.pathname + 
                window.location.search + separator + 'show_chat=true' + 
                window.location.hash;
        }
    }
    
    // Execute on page load if needed
    if (window.location.search.indexOf('show_chat=true') !== -1) {
        console.log('Auto-opening chat from URL param');
        // Set session state directly if possible
        if (window._stState) {
            window._stState.chat_shown = true;
            console.log('Set chat_shown in _stState');
        }
    }
</script>
""", unsafe_allow_html=True)

# In the MAIN area, not dependent on tab rendering, show the chat if enabled
if AI_CHAT_AVAILABLE and st.session_state.get("chat_shown", False):
    try:
        # Use the already initialized BenBot assistant or get from session state
        assistant = benbot_assistant
        if assistant is None and st.session_state.get("benbot_assistant") is not None:
            assistant = st.session_state.benbot_assistant
        
        # Debug logging
        if assistant is not None:
            logger.info("Using BenBot Assistant from main app or session state")
        else:
            logger.warning("BenBot Assistant not available in main scope or session state")
            
        # Add a simple container with the chat widget
        with st.container():
            # Pass the assistant instance to the render function
            try:
                render_ai_chat_widget(assistant)
            except TypeError:
                # If the render function doesn't accept the assistant parameter, call without it
                render_ai_chat_widget()
    except Exception as e:
        logger.error(f"Could not render AI chat: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Could not render AI chat assistant. Check logs for details.")
