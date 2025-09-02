"""
Active Strategies Component

This component displays a live monitor of all active strategies with status and controls.
Uses native Streamlit components for better compatibility.
"""
import streamlit as st
import pandas as pd
import time
import random  # For demo data generation if needed

def render_active_strategies(data_service, api_service, account_type=None):
    """
    Render the active strategies monitor with controls.
    
    Args:
        data_service: Data service for fetching strategy data
        api_service: API service for strategy control actions
        account_type: Optional filter for 'Live' or 'Paper' strategies
    """
    # Add custom CSS for better contrast
    add_dashboard_theme()
    
    # Add CSS for compact strategy cards with dropdowns
    st.markdown("""
    <style>
    /* Compact strategy card styling */
    .compact-card {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: white;
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 5px;
        border: 1px solid #ddd;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .strategy-name {
        font-weight: bold;
        flex-grow: 1;
        color: black !important;
    }
    .badge {
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
        margin-left: 8px;
    }
    .pnl-badge {
        padding: 2px 6px;
        border-radius: 3px;
        font-weight: bold;
        margin-left: 8px;
    }
    /* Better styling for expanders */
    .streamlit-expanderHeader {
        font-weight: normal !important;
        font-size: 0.9em !important;
        color: #555 !important;
    }
    .streamlit-expanderContent {
        background-color: #f9f9f9 !important;
        border-radius: 0 0 5px 5px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    # Get active strategies data
    try:
        strategies_df = data_service.get_active_strategies()
        
        if strategies_df.empty:
            st.warning("No active strategies found.")
            # For demo/development, create some sample data with account type filter
            strategies_df = create_sample_strategies(account_type)
            if strategies_df.empty:  # If still empty, return
                return
    except Exception as e:
        st.error(f"Error fetching strategy data: {str(e)}")
        # For demo/development, create some sample data with account type filter
        strategies_df = create_sample_strategies(account_type)
    
    # Controls section - avoid nesting columns by using a flat layout
    # First row: Filter dropdown and bulk action label
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Custom styling for the filter dropdown to ensure proper contrast
        st.markdown("""
        <style>
        /* Fix the contrast in the filter dropdown */
        div[data-baseweb="select"] {
            background-color: white !important;
        }
        div[data-baseweb="select"] * {
            color: black !important;
        }
        div[data-baseweb="popover"] {
            background-color: white !important;
        }
        div[data-baseweb="popover"] * {
            color: black !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Filter options
        filter_options = ["All Strategies", "Live Only", "Paper Only", "Active Only", "Paused Only"]
        
        filter_choice = st.selectbox(
            "Filter",
            options=filter_options,
            index=0
        )
    
    with col2:
        # Just show the label in this column
        st.write("Bulk Actions:")
    
    # Second row: Bulk action buttons in their own columns (not nested)
    # Create a new separate row of columns for the buttons
    # Custom CSS for blue buttons
    st.markdown("""
    <style>
    div[data-testid="stButton"] button {
        background-color: #0066cc;
        color: white;
        font-weight: 500;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        if st.button("Pause All", help="Pause all filtered strategies"):
            if api_service:
                try:
                    # Get IDs of filtered strategies
                    strategy_ids = strategies_df['id'].tolist()
                    for strategy_id in strategy_ids:
                        api_service.pause_strategy(strategy_id)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error pausing strategies: {str(e)}")
                    
    with btn_col2:
        if st.button("Resume All", help="Resume all filtered strategies"):
            if api_service:
                try:
                    # Get IDs of filtered strategies
                    strategy_ids = strategies_df['id'].tolist()
                    for strategy_id in strategy_ids:
                        api_service.resume_strategy(strategy_id)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error resuming strategies: {str(e)}")
    
    # Fix for the All Strategies dropdown
    st.markdown("""
    <style>
    /* Ensure ALL dropdowns have white background with black text */
    div[data-baseweb="select"] {
        background-color: white !important;
    }
    div[data-baseweb="select"] span {
        color: black !important;
    }
    div[data-baseweb="select"] div {
        background-color: white !important;
    }
    /* Popover menus should always have white background with black text */
    div[data-baseweb="popover"] {
        background-color: white !important;
    }
    div[data-baseweb="popover"] * {
        color: black !important;
    }
    div[data-baseweb="popover"] div {
        background-color: white !important;
    }
    /* Ensure dropdown arrow is visible */
    div[data-baseweb="select"] svg {
        color: black !important;
        fill: black !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Apply filters
    filtered_df = strategies_df.copy()
    
    if filter_choice == "Live Only" and 'phase' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['phase'] == 'LIVE']
    elif filter_choice == "Paper Only" and 'phase' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['phase'] == 'PAPER_TRADE']
    elif filter_choice == "Active Only" and 'status' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['status'] == 'ACTIVE']
    elif filter_choice == "Paused Only" and 'status' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['status'] == 'PAUSED']
    
    if filtered_df.empty:
        st.info(f"No strategies match the '{filter_choice}' filter.")
        return
    
    # Create strategy cards
    for index, row in filtered_df.iterrows():
        strategy_id = row.get('id', f"strategy_{index}")
        strategy_name = row.get('name', 'Unnamed Strategy')
        status = row.get('status', 'UNKNOWN')
        phase = row.get('phase', 'UNKNOWN')
        daily_pnl = row.get('daily_pnl', 0.0)
        total_pnl = row.get('total_pnl', 0.0)
        
        # Status and phase color mapping
        status_colors = {
            'ACTIVE': ('#CCFFCC', '#006600'),  # Light green bg, dark green text
            'PAUSED': ('#FFE8CC', '#663300'),  # Light orange bg, dark orange text
            'ERROR': ('#FFCCCC', '#990000'),   # Light red bg, dark red text
            'STOPPED': ('#CCCCCC', '#333333'), # Light gray bg, dark gray text
            'UNKNOWN': ('#EAEAEA', '#666666')  # Light gray bg, darker gray text
        }
        phase_colors = {
            'LIVE': ('#D1F0FF', '#003366'),      # Light blue bg, dark blue text
            'PAPER_TRADE': ('#E6E6FF', '#000066'), # Light purple bg, dark purple text
            'BACKTEST': ('#E6FFE6', '#003300'),    # Light green bg, dark green text
            'UNKNOWN': ('#EAEAEA', '#666666')      # Light gray bg, darker gray text
        }
        
        # Get colors for current status and phase
        status_bg_color, status_text_color = status_colors.get(status, status_colors['UNKNOWN'])
        phase_bg_color, phase_text_color = phase_colors.get(phase, phase_colors['UNKNOWN'])
        
        # Daily P&L indicator colors
        pnl_bg_color = '#CCFFCC' if daily_pnl >= 0 else '#FFCCCC'
        pnl_text_color = '#006600' if daily_pnl >= 0 else '#990000'
        
        # Create the compact header with name, status, phase and daily P&L
        header_html = f"""
        <div class="compact-card">
            <div class="strategy-name">{strategy_name}</div>
            <span class="badge" style="background-color: {status_bg_color}; color: {status_text_color};">{status}</span>
            <span class="badge" style="background-color: {phase_bg_color}; color: {phase_text_color};">{phase.replace('_', ' ')}</span>
            <span class="pnl-badge" style="background-color: {pnl_bg_color}; color: {pnl_text_color};">Daily: ${daily_pnl:.2f}</span>
        </div>
        """
        
        # Display the header
        st.markdown(header_html, unsafe_allow_html=True)
        
        # Create an expander for the details
        with st.expander("Details & Actions"):
            # Details section
            col1, col2 = st.columns(2)
            with col1:
                # Total P&L indicator with proper contrast
                total_pnl_bg_color = '#CCFFCC' if total_pnl >= 0 else '#FFCCCC'
                total_pnl_text_color = '#006600' if total_pnl >= 0 else '#990000'
                st.markdown(f"""<div>Total P&L: <span style="background-color: {total_pnl_bg_color}; color: {total_pnl_text_color}; 
                                     padding: 2px 6px; border-radius: 3px; font-weight: bold;">
                                    ${total_pnl:.2f}
                                 </span></div>""", unsafe_allow_html=True)
                
                st.markdown(f"**Win Rate:** {row.get('win_rate', 0):.2f}%")
            
            with col2:
                st.markdown(f"**Trades Today:** {row.get('trades_today', 0)}")
                st.markdown(f"**Positions:** {row.get('positions', 0)}")
            
            # Action buttons section
            st.markdown("<hr style='margin: 10px 0; border-color: #eee;'>", unsafe_allow_html=True)
            
            col_left, col_right = st.columns(2)
            with col_left:
                # Pause/Resume button
                if status == 'ACTIVE':
                    if st.button("Pause", key=f"pause_{strategy_id}", help="Pause this strategy"):
                        if api_service:
                            try:
                                api_service.pause_strategy(strategy_id)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error pausing strategy: {str(e)}")
                else:  # status == 'PAUSED'
                    if st.button("Resume", key=f"resume_{strategy_id}", help="Resume this strategy"):
                        if api_service:
                            try:
                                api_service.resume_strategy(strategy_id)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error resuming strategy: {str(e)}")  
            
            with col_right:
                # Close positions button
                if st.button("Close", key=f"close_{strategy_id}", help="Close positions"):
                    if api_service:
                        try:
                            api_service.close_strategy_positions(strategy_id)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error closing positions: {str(e)}")

# Add helper function to generate sample data if needed
# Add custom CSS for dashboard theme with maximum contrast (opposite colors)
def add_dashboard_theme():
    """Add custom CSS to ensure opposite colors for background and text for maximum contrast"""
    st.markdown("""
    <style>
    /* ===== MAIN DASHBOARD STYLING: CONTRAST OPTIMIZED ===== */
    /* Base: Light gray background with dark text */
    .stApp {
        background-color: #f5f5f5;
        color: #111;
    }
    
    /* Metrics: White cards with black text */
    [data-testid="stMetric"] {
        background-color: white;
        padding: 10px 15px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.15);
        border: 1px solid #ddd;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: black !important;
    }
    
    /* Buttons: Blue with white text */
    div[data-testid="stButton"] button, .stButton>button {
        background-color: #0066cc !important;
        color: white !important;
        font-weight: 500 !important;
        border: none !important;
    }
    
    /* Dropdowns & Selects: White with black text */
    div[data-baseweb="select"], .stSelectbox [data-baseweb="select"] {
        background-color: white !important;
        border: 1px solid #ddd !important;
    }
    div[data-baseweb="select"] *, .stSelectbox [data-baseweb="select"] * {
        color: black !important;
    }
    div[data-baseweb="popover"], .stSelectbox [data-baseweb="popover"] {
        background-color: white !important;
    }
    div[data-baseweb="popover"] *, .stSelectbox [data-baseweb="popover"] * {
        color: black !important;
    }
    
    /* Sidebar: Dark with white text */
    [data-testid="stSidebar"] {
        background-color: #222;
    }
    
    /* Typography: Ensure proper contrast throughout */
    h1, h2, h3, .stMarkdown, .stText, .stTextArea, .stTextInput {
        color: black !important;
    }
    h1, h2, h3 {
        font-weight: 600;
    }
    
    /* Strategy cards: White with black text + colored accents */
    [data-testid="stVerticalBlock"] > div:nth-of-type(1) {
        border-left: 4px solid #4CAF50;
        background-color: white;
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        border: 1px solid #ddd;
    }
    [data-testid="stVerticalBlock"] > div:nth-of-type(1) * {
        color: #333 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def create_sample_strategies(account_type=None):
    """Create sample strategy data for development and testing
    
    Args:
        account_type: Optional filter for 'LIVE' or 'PAPER_TRADE' strategies
    """
    # Create an empty DataFrame with the expected columns
    columns = [
        'id', 'name', 'phase', 'status', 'daily_pnl', 'total_pnl',
        'win_rate', 'trades_today', 'positions', 'type', 'source', 
        'creation_date', 'optimization_level', 'description'
    ]
    
    # Create comprehensive sample data that reflects the sophisticated trading system
    data = [
        # LIVE STRATEGIES - CRYPTO
        {
            'id': 'grid_btc_01',
            'name': 'BTC Grid Strategy',
            'phase': 'LIVE',
            'status': 'ACTIVE',
            'daily_pnl': 145.75,
            'total_pnl': 2876.50,
            'win_rate': 68.5,
            'trades_today': 8,
            'positions': 5,
            'type': 'CRYPTO',
            'source': 'Quantitative',
            'creation_date': '2025-03-15',
            'optimization_level': 'High',
            'description': 'Advanced grid trading strategy for Bitcoin with dynamic grid placement based on volatility'
        },
        {
            'id': 'eth_momentum_02',
            'name': 'ETH Momentum',
            'phase': 'LIVE',
            'status': 'ACTIVE',
            'daily_pnl': 78.35,
            'total_pnl': 1432.90,
            'win_rate': 64.2,
            'trades_today': 3,
            'positions': 2,
            'type': 'CRYPTO',
            'source': 'Technical',
            'creation_date': '2025-02-28',
            'optimization_level': 'Medium',
            'description': 'Ethereum momentum strategy using MACD and RSI with volatility-based position sizing'
        },
        
        # LIVE STRATEGIES - FOREX
        {
            'id': 'forex_gbpusd_rsi',
            'name': 'GBP/USD RSI Edge',
            'phase': 'LIVE',
            'status': 'ACTIVE',
            'daily_pnl': 112.85,
            'total_pnl': 3456.78,
            'win_rate': 72.4,
            'trades_today': 5,
            'positions': 1,
            'type': 'FOREX',
            'source': 'Technical',
            'creation_date': '2025-01-10',
            'optimization_level': 'High',
            'description': 'GBP/USD strategy using RSI divergence with news sentiment adjustment from Cinnamon NLP'
        },
        {
            'id': 'forex_eurusd_news',
            'name': 'EUR/USD News Reactor',
            'phase': 'LIVE',
            'status': 'PAUSED',
            'daily_pnl': 0.00,
            'total_pnl': 984.25,
            'win_rate': 68.1,
            'trades_today': 0,
            'positions': 0,
            'type': 'FOREX',
            'source': 'NLP/News',
            'creation_date': '2025-03-05',
            'optimization_level': 'High',
            'description': 'NLP-based strategy reacting to ECB and Fed news with Cinnamon sentiment analysis'
        },
        
        # LIVE STRATEGIES - STOCKS
        {
            'id': 'aapl_nlp_earnings',
            'name': 'AAPL Earnings Edge',
            'phase': 'LIVE',
            'status': 'ACTIVE',
            'daily_pnl': 235.45,
            'total_pnl': 5632.90,
            'win_rate': 75.2,
            'trades_today': 1,
            'positions': 1,
            'type': 'STOCK',
            'source': 'NLP/News',
            'creation_date': '2025-02-01',
            'optimization_level': 'Very High',
            'description': 'Apple earnings strategy with NLP analysis of conference calls and social sentiment'
        },
        {
            'id': 'tsla_twitter_nlp',
            'name': 'TSLA Twitter Sentiment',
            'phase': 'LIVE',
            'status': 'ACTIVE',
            'daily_pnl': -125.50,
            'total_pnl': 3245.75,
            'win_rate': 62.8,
            'trades_today': 2,
            'positions': 1,
            'type': 'STOCK',
            'source': 'NLP/Social',
            'creation_date': '2025-01-22',
            'optimization_level': 'High',
            'description': 'Tesla strategy using Cinnamon NLP to analyze Twitter sentiment with technical confirmation'
        },
        {
            'id': 'amzn_news_breakout',
            'name': 'AMZN News Breakout',
            'phase': 'LIVE',
            'status': 'ACTIVE',
            'daily_pnl': 89.25,
            'total_pnl': 1876.50,
            'win_rate': 64.5,
            'trades_today': 1,
            'positions': 1,
            'type': 'STOCK',
            'source': 'NLP/Technical',
            'creation_date': '2025-03-18',
            'optimization_level': 'High',
            'description': 'Amazon breakout strategy triggered by news sentiment analysis and technical confirmation'
        },
        {
            'id': 'energy_sector_rotation',
            'name': 'Energy Sector Rotation',
            'phase': 'LIVE',
            'status': 'ACTIVE',
            'daily_pnl': 312.45,
            'total_pnl': 4532.75,
            'win_rate': 68.9,
            'trades_today': 3,
            'positions': 2,
            'type': 'STOCK',
            'source': 'Fundamental/NLP',
            'creation_date': '2025-02-15',
            'optimization_level': 'Medium',
            'description': 'Energy sector rotation strategy using NLP analysis of news and earnings reports'
        },
        
        # PAPER TRADING STRATEGIES - CRYPTO
        {
            'id': 'sol_sentiment_vol',
            'name': 'SOL Sentiment Volatility',
            'phase': 'PAPER_TRADE',
            'status': 'ACTIVE',
            'daily_pnl': 74.25,
            'total_pnl': 625.50,
            'win_rate': 59.8,
            'trades_today': 4,
            'positions': 1,
            'type': 'CRYPTO',
            'source': 'NLP/Technical',
            'creation_date': '2025-04-05',
            'optimization_level': 'Medium',
            'description': 'Solana strategy combining social sentiment analysis with volatility breakout patterns'
        },
        {
            'id': 'multi_crypto_arb',
            'name': 'Multi-Exchange Arbitrage',
            'phase': 'PAPER_TRADE',
            'status': 'ACTIVE',
            'daily_pnl': 45.75,
            'total_pnl': 328.90,
            'win_rate': 82.5,
            'trades_today': 12,
            'positions': 3,
            'type': 'CRYPTO',
            'source': 'Quantitative',
            'creation_date': '2025-04-10',
            'optimization_level': 'Very High',
            'description': 'Cross-exchange arbitrage strategy for BTC, ETH and SOL with automated execution'
        },
        
        # PAPER TRADING STRATEGIES - FOREX
        {
            'id': 'gbpjpy_nlp_technical',
            'name': 'GBP/JPY News-Tech Hybrid',
            'phase': 'PAPER_TRADE',
            'status': 'ACTIVE',
            'daily_pnl': -25.45,
            'total_pnl': 175.30,
            'win_rate': 54.2,
            'trades_today': 2,
            'positions': 1,
            'type': 'FOREX',
            'source': 'NLP/Technical',
            'creation_date': '2025-04-18',
            'optimization_level': 'Medium',
            'description': 'GBP/JPY strategy combining BoE/BoJ news sentiment with technical breakout patterns'
        },
        
        # PAPER TRADING STRATEGIES - STOCKS
        {
            'id': 'ai_sector_momentum',
            'name': 'AI Sector Momentum',
            'phase': 'PAPER_TRADE',
            'status': 'ACTIVE',
            'daily_pnl': 95.25,
            'total_pnl': 450.75,
            'win_rate': 61.8,
            'trades_today': 3,
            'positions': 2,
            'type': 'STOCK',
            'source': 'NLP/Technical',
            'creation_date': '2025-04-02',
            'optimization_level': 'High',
            'description': 'AI sector momentum strategy using NLP analysis of news and earnings with technical filters'
        },
        {
            'id': 'ev_supply_chain',
            'name': 'EV Supply Chain',
            'phase': 'PAPER_TRADE',
            'status': 'PAUSED',
            'daily_pnl': 0.00,
            'total_pnl': 215.80,
            'win_rate': 58.2,
            'trades_today': 0,
            'positions': 0,
            'type': 'STOCK',
            'source': 'Fundamental/NLP',
            'creation_date': '2025-03-28',
            'optimization_level': 'Medium',
            'description': 'EV supply chain analysis strategy using NLP to identify key suppliers mentioned in news'
        },
        {
            'id': 'semiconductor_earnings',
            'name': 'Semiconductor Earnings',
            'phase': 'PAPER_TRADE',
            'status': 'ACTIVE',
            'daily_pnl': 67.85,
            'total_pnl': 398.50,
            'win_rate': 65.3,
            'trades_today': 2,
            'positions': 2,
            'type': 'STOCK',
            'source': 'NLP/Earnings',
            'creation_date': '2025-04-12',
            'optimization_level': 'High',
            'description': 'Semiconductor sector strategy analyzing earnings calls and guidance using Cinnamon NLP'
        }
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Filter by account type if specified
    if account_type == 'Live':
        df = df[df['phase'] == 'LIVE']
    elif account_type == 'Paper':
        df = df[df['phase'] == 'PAPER_TRADE']
    
    return df
