#!/usr/bin/env python3
"""
Live Trading Dashboard

A Streamlit-based dashboard for monitoring trading system performance,
allocations, and trading decisions in real-time.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import time
import threading
import queue
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import local modules
from trading_bot.data.real_time_data_processor import RealTimeDataManager
from trading_bot.optimization.advanced_market_regime_detector import AdvancedMarketRegimeDetector
from trading_bot.risk_manager import RiskManager  # Import the risk manager
from trading_bot.assistant.benbot_assistant import BenBotAssistant  # Import the AI assistant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define color schemes for different market regimes
REGIME_COLORS = {
    'bull': '#4CAF50',     # Green
    'bear': '#F44336',     # Red
    'consolidation': '#2196F3',  # Blue
    'volatility': '#FF9800',  # Orange
    'recovery': '#9C27B0',  # Purple
    'unknown': '#9E9E9E'   # Grey
}

# Global state variables
class DashboardState:
    """Class to store global dashboard state that persists between Streamlit reruns"""
    def __init__(self):
        self.data_queue = queue.Queue()
        self.portfolio_history = deque(maxlen=1000)  # Store up to 1000 points
        self.trade_history = deque(maxlen=100)  # Store last 100 trades
        self.market_regime_history = deque(maxlen=100)  # Store last 100 regime changes
        self.last_update_time = datetime.now()
        self.streaming_active = False
        self.data_manager = None
        self.symbols = []
        
        # Risk management data
        self.risk_metrics = {
            "portfolio_risk": 32.0,
            "drawdown": 8.2,
            "var_95": 2450.0,
            "max_drawdown": 15.0,
            "leverage": 1.2,
            "position_size_modifier": 1.0,
            "psychological_risk_score": 42.0,
            "risk_level": "Moderate"
        }
        self.risk_exposure = {}
        self.position_limits = {}
        self.correlation_matrix = None
        self.high_correlation_pairs = []
        self.risk_heat_map = {}
        self.psychological_patterns = []
        self.risk_recommendations = []
        
        # AI Assistant data
        self.ai_assistant = None
        self.chat_history = []
        self.strategy_recommendations = []

state = DashboardState()

def initialize_dashboard():
    """Configure the dashboard layout and settings"""
    st.set_page_config(
        page_title="Live Trading Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Set page header
    st.title("Live Trading Dashboard")
    st.markdown("Real-time monitoring of trading performance, allocations, and decisions")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Data source selection
        data_source = st.selectbox(
            "Data Source", 
            ["Alpaca", "Interactive Brokers", "Mock Data"],
            index=0
        )
        
        # Symbol configuration
        default_symbols = "SPY,QQQ,IWM,GLD"
        symbols_input = st.text_input("Symbols (comma-separated)", default_symbols)
        symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
        
        # Update frequency
        update_frequency = st.slider(
            "Update Frequency (seconds)", 
            min_value=1, 
            max_value=60, 
            value=5
        )
        
        # Time range for historical data
        time_range = st.selectbox(
            "Time Range",
            ["1 Hour", "4 Hours", "1 Day", "1 Week", "1 Month"],
            index=2
        )
        
        # Connect button
        connect_button = st.button("Connect" if not state.streaming_active else "Disconnect")
        
        if connect_button:
            if state.streaming_active:
                stop_data_streaming()
            else:
                start_data_streaming(symbols, data_source)
        
        # Display connection status
        connection_status = "Connected" if state.streaming_active else "Disconnected"
        status_color = "#4CAF50" if state.streaming_active else "#F44336"
        st.markdown(f"<h3 style='color: {status_color};'>Status: {connection_status}</h3>", unsafe_allow_html=True)
        
        # Display last update time
        st.write(f"Last Update: {state.last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # About section
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard connects to real-time market data and visualizes 
        trading performance, allocations, and decisions for your trading system.
        
        Use the controls above to configure the dashboard.
        """)
    
    return symbols, update_frequency, time_range, data_source

def start_data_streaming(symbols, data_source):
    """Start streaming market data"""
    if state.streaming_active:
        return
    
    state.symbols = symbols
    logger.info(f"Starting data streaming for symbols: {symbols}")
    
    try:
        # Configure data manager based on selected data source
        config = {
            'data_source': data_source.lower().replace(" ", "_"),
            'timeframes': ['1min', '5min', '15min', '1hour', '1day'],
            'use_market_regimes': True,
        }
        
        # Add credentials for the selected data source
        if data_source == "Alpaca":
            config['alpaca_config'] = {
                'api_key': os.environ.get('ALPACA_API_KEY', 'demo-api-key'),
                'api_secret': os.environ.get('ALPACA_API_SECRET', 'demo-api-secret')
            }
        elif data_source == "Interactive Brokers":
            config['ib_config'] = {
                'host': os.environ.get('IB_HOST', '127.0.0.1'),
                'port': int(os.environ.get('IB_PORT', '7497')),
                'client_id': int(os.environ.get('IB_CLIENT_ID', '1'))
            }
        
        # Initialize data manager
        if data_source == "Mock Data":
            # Create mock data instead of connecting to real data source
            create_mock_data_thread(symbols, state.data_queue)
        else:
            state.data_manager = RealTimeDataManager(symbols, config)
            
            # Register callbacks
            state.data_manager.on_bar_update = on_bar_update
            state.data_manager.on_regime_change = on_regime_change
            state.data_manager.on_strategy_update = on_strategy_update
            
            # Start the data manager
            state.data_manager.start()
        
        # Initialize risk manager connection if possible
        connect_to_risk_manager(data_source)
        
        # Initialize AI assistant
        initialize_ai_assistant(data_source)
        
        state.streaming_active = True
        state.last_update_time = datetime.now()
        
        # Initialize portfolio history with some starting data
        if len(state.portfolio_history) == 0:
            state.portfolio_history.append({
                'timestamp': datetime.now(),
                'value': 100000.0,
                'cash': 100000.0,
                'invested': 0.0
            })
        
        logger.info("Data streaming started successfully")
        
    except Exception as e:
        logger.error(f"Error starting data streaming: {str(e)}")
        st.error(f"Failed to start data streaming: {str(e)}")

def stop_data_streaming():
    """Stop streaming market data"""
    if not state.streaming_active:
        return
    
    logger.info("Stopping data streaming")
    
    try:
        if state.data_manager:
            state.data_manager.stop()
            state.data_manager = None
        
        state.streaming_active = False
        logger.info("Data streaming stopped successfully")
        
    except Exception as e:
        logger.error(f"Error stopping data streaming: {str(e)}")
        st.error(f"Failed to stop data streaming: {str(e)}")

def on_bar_update(data):
    """Handle new bar data from the data manager"""
    # Add data to queue for processing
    state.data_queue.put({
        'type': 'bar_update',
        'data': data,
        'timestamp': datetime.now()
    })
    
    # Update portfolio value (in a real system, this would use actual portfolio data)
    if datetime.now().second % 5 == 0:  # Update every 5 seconds to reduce computation
        update_mock_portfolio()
        
        # Also update risk metrics periodically
        update_mock_risk_metrics()

def on_regime_change(regime):
    """Handle market regime changes from the data manager"""
    # Add data to queue for processing
    state.data_queue.put({
        'type': 'regime_change',
        'regime': regime,
        'timestamp': datetime.now()
    })
    
    # Add to regime history
    state.market_regime_history.append({
        'timestamp': datetime.now(),
        'regime': regime,
        'duration': 0  # To be calculated
    })
    
    logger.info(f"Market regime changed to: {regime}")

def on_strategy_update(weights):
    """Handle strategy weight updates from the data manager"""
    # Add data to queue for processing
    state.data_queue.put({
        'type': 'strategy_update',
        'weights': weights,
        'timestamp': datetime.now()
    })
    
    # Simulate a trade based on the new weights
    simulate_trade(weights)
    
    logger.info(f"Strategy weights updated: {weights}")

def update_mock_portfolio():
    """Update mock portfolio data for demonstration purposes"""
    if not state.portfolio_history:
        return
    
    last_value = state.portfolio_history[-1]['value']
    
    # Generate random portfolio change
    random_change = np.random.normal(0, 0.001)  # Small random fluctuation
    
    # Add trend based on market regime
    current_regime = "unknown"
    if state.market_regime_history:
        current_regime = state.market_regime_history[-1]['regime']
    
    regime_factor = {
        'bull': 0.0002,
        'bear': -0.0002,
        'consolidation': 0.0,
        'volatility': np.random.choice([-0.0004, 0.0004]),
        'recovery': 0.0001,
        'unknown': 0.0
    }.get(current_regime, 0.0)
    
    # Calculate new portfolio value
    new_value = last_value * (1 + random_change + regime_factor)
    
    # Calculate cash and invested amounts (mockup)
    total_invested_percent = min(0.8, len(state.trade_history) * 0.1)  # Max 80% invested
    invested = new_value * total_invested_percent
    cash = new_value - invested
    
    # Add to portfolio history
    state.portfolio_history.append({
        'timestamp': datetime.now(),
        'value': new_value,
        'cash': cash,
        'invested': invested
    })
    
    state.last_update_time = datetime.now()

def simulate_trade(weights):
    """Simulate a trade based on strategy weights for demonstration purposes"""
    # Calculate a random symbol from available symbols
    if not state.symbols:
        return
    
    symbol = np.random.choice(state.symbols)
    
    # Determine if it's a buy or sell
    action = np.random.choice(['BUY', 'SELL'])
    
    # Generate random quantity and price
    quantity = np.random.randint(10, 100)
    price = np.random.uniform(100, 500)
    
    # Calculate trade value
    value = quantity * price
    
    # Add to trade history
    state.trade_history.append({
        'timestamp': datetime.now(),
        'symbol': symbol,
        'action': action,
        'quantity': quantity,
        'price': price,
        'value': value,
        'weights': weights
    })
    
    logger.info(f"Simulated trade: {action} {quantity} {symbol} @ {price}")

def create_mock_data_thread(symbols, data_queue):
    """Create a thread that generates mock data for demonstration purposes"""
    def mock_data_generator():
        """Generate mock market data"""
        regimes = ['bull', 'bear', 'consolidation', 'volatility', 'recovery']
        current_regime = np.random.choice(regimes)
        last_regime_change = datetime.now()
        
        # Add initial regime
        on_regime_change(current_regime)
        
        # Generate mock prices for each symbol
        prices = {symbol: np.random.uniform(100, 500) for symbol in symbols}
        
        while state.streaming_active:
            try:
                current_time = datetime.now()
                
                # Occasionally change market regime
                if (current_time - last_regime_change).total_seconds() > np.random.randint(60, 180):
                    current_regime = np.random.choice(regimes)
                    last_regime_change = current_time
                    on_regime_change(current_regime)
                
                # Update prices based on regime
                for symbol in symbols:
                    # Base random movement
                    random_change = np.random.normal(0, 0.002)
                    
                    # Add regime bias
                    regime_factor = {
                        'bull': 0.0005,
                        'bear': -0.0005,
                        'consolidation': 0.0,
                        'volatility': np.random.choice([-0.001, 0.001]),
                        'recovery': 0.0003
                    }[current_regime]
                    
                    # Update price
                    prices[symbol] *= (1 + random_change + regime_factor)
                    
                    # Create mock bar data
                    bar_data = {
                        'symbol': symbol,
                        'timestamp': current_time,
                        'price': prices[symbol],
                        'volume': np.random.randint(1000, 10000),
                        'type': 'bar'
                    }
                    
                    # Send to update handler
                    on_bar_update(bar_data)
                
                # Occasionally update strategy weights
                if np.random.random() < 0.05:  # 5% chance each iteration
                    strategy_names = ['MA_Trend', 'Mean_Reversion', 'Momentum', 'Volatility_Breakout']
                    weights = {name: np.random.random() for name in strategy_names}
                    # Normalize to sum to 1
                    total = sum(weights.values())
                    weights = {k: v/total for k, v in weights.items()}
                    
                    on_strategy_update(weights)
                
                # Simulate portfolio update
                update_mock_portfolio()
                
                time.sleep(1)  # Generate data every second
                
            except Exception as e:
                logger.error(f"Error in mock data generator: {str(e)}")
                time.sleep(1)
    
    # Start the mock data generator in a background thread
    thread = threading.Thread(target=mock_data_generator, daemon=True)
    thread.start()
    logger.info("Started mock data generator")

def update_mock_risk_metrics():
    """Update mock risk metrics for demonstration purposes"""
    # In a real implementation, this would fetch data from the risk manager
    
    # Generate small random changes to risk metrics
    risk_delta = np.random.normal(0, 0.5)
    dd_delta = np.random.normal(0, 0.2)
    var_delta = np.random.normal(0, 50)
    
    # Update stored risk metrics
    state.risk_metrics["portfolio_risk"] = max(0, min(100, state.risk_metrics["portfolio_risk"] + risk_delta))
    state.risk_metrics["drawdown"] = max(0, min(state.risk_metrics["max_drawdown"], state.risk_metrics["drawdown"] + dd_delta))
    state.risk_metrics["var_95"] = max(0, state.risk_metrics["var_95"] + var_delta)
    
    # Update risk level based on portfolio risk
    if state.risk_metrics["portfolio_risk"] < 20:
        state.risk_metrics["risk_level"] = "Low"
    elif state.risk_metrics["portfolio_risk"] < 40:
        state.risk_metrics["risk_level"] = "Moderate"
    elif state.risk_metrics["portfolio_risk"] < 60:
        state.risk_metrics["risk_level"] = "Elevated"
    elif state.risk_metrics["portfolio_risk"] < 80:
        state.risk_metrics["risk_level"] = "High"
    else:
        state.risk_metrics["risk_level"] = "Extreme"
    
    # Update psychological risk metrics occasionally
    if np.random.random() < 0.2:  # 20% chance
        psych_delta = np.random.normal(0, 3)
        state.risk_metrics["psychological_risk_score"] = max(0, min(100, state.risk_metrics["psychological_risk_score"] + psych_delta))
        
        # Update position size modifier based on psychological risk
        psych_score = state.risk_metrics["psychological_risk_score"]
        if psych_score <= 20:
            state.risk_metrics["position_size_modifier"] = 1.0
        elif psych_score >= 80:
            state.risk_metrics["position_size_modifier"] = 0.25
        else:
            state.risk_metrics["position_size_modifier"] = 1.0 - (psych_score - 20) / 80 * 0.75

def check_data_updates():
    """Process any incoming data updates from the queue"""
    if not state.streaming_active:
        return
    
    try:
        # Process up to 10 updates at a time to avoid blocking
        for _ in range(10):
            try:
                # Get the next update with a small timeout
                update = state.data_queue.get(block=False)
                
                # Process different update types
                if update['type'] == 'bar_update':
                    # Process bar data updates
                    # In a real implementation, this would update price charts
                    pass
                    
                elif update['type'] == 'regime_change':
                    # Process market regime change
                    # Handled by on_regime_change function
                    pass
                    
                elif update['type'] == 'strategy_update':
                    # Process strategy weight updates
                    # Handled by on_strategy_update function
                    pass
                    
                elif update['type'] == 'risk_update':
                    # Process risk metric updates
                    if 'metrics' in update:
                        state.risk_metrics.update(update['metrics'])
                    
                    if 'exposure' in update:
                        state.risk_exposure = update['exposure']
                    
                    if 'limits' in update:
                        state.position_limits = update['limits']
                    
                    if 'correlation' in update:
                        state.correlation_matrix = update['correlation']
                        
                        # Update highly correlated pairs
                        if 'high_correlations' in update:
                            state.high_correlation_pairs = update['high_correlations']
                    
                    if 'psychological' in update:
                        state.risk_metrics["psychological_risk_score"] = update['psychological'].get('score', 
                            state.risk_metrics["psychological_risk_score"])
                        state.psychological_patterns = update['psychological'].get('patterns', [])
                        state.risk_recommendations = update['psychological'].get('recommendations', [])
                
                # Mark as processed
                state.data_queue.task_done()
                
            except queue.Empty:
                # No more updates in the queue
                break
    
    except Exception as e:
        logger.error(f"Error processing data updates: {str(e)}")

def display_portfolio_performance():
    """Display portfolio performance charts"""
    st.header("Portfolio Performance")
    
    if not state.portfolio_history:
        st.info("No portfolio data available. Connect to a data source to begin.")
        return
    
    # Convert portfolio history to DataFrame
    df = pd.DataFrame(list(state.portfolio_history))
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Portfolio Value", "Allocation", "Performance Metrics"])
    
    with tab1:
        # Portfolio value over time
        fig = px.line(
            df, 
            x='timestamp', 
            y='value',
            title='Portfolio Value Over Time',
            line_shape='spline'
        )
        
        # Add cash and invested areas
        if 'cash' in df.columns and 'invested' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['invested'],
                    fill='tozeroy',
                    name='Invested',
                    line=dict(color='rgba(0, 128, 0, 0.7)')
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['cash'],
                    fill='tonexty',
                    name='Cash',
                    line=dict(color='rgba(0, 0, 255, 0.7)')
                )
            )
        
        # Format the chart
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Value ($)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Current allocation pie chart
        if len(df) > 0 and 'cash' in df.columns and 'invested' in df.columns:
            last_row = df.iloc[-1]
            
            # Create a pie chart of current allocation
            labels = ['Cash', 'Invested']
            values = [last_row['cash'], last_row['invested']]
            
            fig = px.pie(
                values=values,
                names=labels,
                title='Current Portfolio Allocation',
                color_discrete_sequence=['rgba(0, 0, 255, 0.7)', 'rgba(0, 128, 0, 0.7)']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # If we have trade data, show allocation by symbol
            if state.trade_history:
                # Calculate position by symbol
                positions = {}
                for trade in state.trade_history:
                    symbol = trade['symbol']
                    value = trade['value']
                    action = trade['action']
                    
                    if symbol not in positions:
                        positions[symbol] = 0
                    
                    if action == 'BUY':
                        positions[symbol] += value
                    else:  # SELL
                        positions[symbol] -= value
                
                # Filter out closed positions
                positions = {k: v for k, v in positions.items() if v > 0}
                
                if positions:
                    # Create a pie chart of symbol allocation
                    fig = px.pie(
                        values=list(positions.values()),
                        names=list(positions.keys()),
                        title='Allocation by Symbol'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Calculate performance metrics
        if len(df) > 1:
            # Calculate returns
            df['return'] = df['value'].pct_change()
            
            # Calculate cumulative return
            first_value = df['value'].iloc[0]
            last_value = df['value'].iloc[-1]
            total_return = (last_value / first_value) - 1
            
            # Calculate annualized metrics (assuming minutely data)
            minutes = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 60
            annualization_factor = np.sqrt(525600 / max(minutes, 1))  # 525600 minutes in a year
            
            volatility = df['return'].std() * annualization_factor
            sharpe = (df['return'].mean() / df['return'].std()) * annualization_factor if df['return'].std() > 0 else 0
            
            # Calculate drawdown
            df['cumulative_return'] = (1 + df['return']).cumprod()
            df['running_max'] = df['cumulative_return'].cummax()
            df['drawdown'] = df['cumulative_return'] / df['running_max'] - 1
            max_drawdown = df['drawdown'].min()
            
            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", f"{total_return:.2%}")
                
            with col2:
                st.metric("Volatility (Ann.)", f"{volatility:.2%}")
                
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                
            with col4:
                st.metric("Max Drawdown", f"{max_drawdown:.2%}")
            
            # Display drawdown chart
            fig = px.area(
                df, 
                x='timestamp', 
                y='drawdown',
                title='Portfolio Drawdown',
                color_discrete_sequence=['rgba(255, 0, 0, 0.5)']
            )
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Drawdown",
                hovermode="x unified",
                yaxis=dict(
                    tickformat=".1%",
                    range=[min(min(df['drawdown']*1.1, -0.01), -0.05), 0.01]
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display rolling performance metrics
            if len(df) > 20:
                # Calculate rolling metrics
                window = min(20, len(df) // 2)
                df['rolling_return'] = df['return'].rolling(window=window).mean() * window
                df['rolling_volatility'] = df['return'].rolling(window=window).std() * np.sqrt(window)
                df['rolling_sharpe'] = df['rolling_return'] / df['rolling_volatility']
                
                # Create subplots
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=["Rolling Return", "Rolling Sharpe Ratio"],
                    vertical_spacing=0.12
                )
                
                # Add rolling return
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['rolling_return'],
                        name='Rolling Return',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                
                # Add rolling Sharpe
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['rolling_sharpe'],
                        name='Rolling Sharpe',
                        line=dict(color='green')
                    ),
                    row=2, col=1
                )
                
                # Format the chart
                fig.update_layout(
                    height=400,
                    hovermode="x unified",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)

def display_market_data():
    """Display real-time market data and price charts"""
    st.header("Market Data")
    
    if not state.streaming_active:
        st.info("Connect to a data source to view real-time market data.")
        return
    
    # Create tabs for different symbols
    if state.symbols:
        tabs = st.tabs(state.symbols)
        
        for i, symbol in enumerate(state.symbols):
            with tabs[i]:
                # Price chart for the symbol
                if state.data_manager:
                    # Get actual price data if available
                    df = state.data_manager.get_latest_bars(symbol, '1min', 100)
                    if not df.empty:
                        # Create candlestick chart
                        fig = go.Figure(data=[go.Candlestick(
                            x=df.index,
                            open=df['open'],
                            high=df['high'],
                            low=df['low'],
                            close=df['close'],
                            name='Price'
                        )])
                        
                        # Add volume as a bar chart
                        if 'volume' in df.columns:
                            fig.add_trace(go.Bar(
                                x=df.index,
                                y=df['volume'],
                                name='Volume',
                                marker_color='rgba(0, 0, 255, 0.3)',
                                yaxis="y2"
                            ))
                            
                            fig.update_layout(
                                yaxis2=dict(
                                    title="Volume",
                                    overlaying="y",
                                    side="right",
                                    showgrid=False
                                )
                            )
                        
                        fig.update_layout(
                            title=f"{symbol} Price Chart",
                            xaxis_title="Time",
                            yaxis_title="Price",
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No price data available for {symbol}.")
                else:
                    # Show mock price chart for demonstration
                    if state.portfolio_history:
                        # Generate mock price data based on timestamp from portfolio data
                        timestamps = [entry['timestamp'] for entry in state.portfolio_history]
                        
                        # Generate random price with some correlation to portfolio
                        base_price = 100
                        price_data = []
                        
                        for i, ts in enumerate(timestamps):
                            if i == 0:
                                price = base_price
                            else:
                                # Get portfolio return
                                port_return = state.portfolio_history[i]['value'] / state.portfolio_history[i-1]['value'] - 1
                                
                                # Add correlated movement plus noise
                                correlation = 0.7 if symbol in ['SPY', 'QQQ'] else 0.3
                                price_return = correlation * port_return + (1-correlation) * np.random.normal(0, 0.001)
                                price = price_data[-1] * (1 + price_return)
                            
                            price_data.append(price)
                        
                        # Create DataFrame
                        df = pd.DataFrame({
                            'timestamp': timestamps,
                            'price': price_data
                        })
                        
                        # Create line chart
                        fig = px.line(
                            df, 
                            x='timestamp', 
                            y='price',
                            title=f"{symbol} Price (Mock Data)",
                            line_shape='spline'
                        )
                        
                        fig.update_layout(
                            xaxis_title="Time",
                            yaxis_title="Price",
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No price data available for {symbol}.")

def display_market_regime():
    """Display market regime information"""
    st.header("Market Regime Analysis")
    
    if not state.market_regime_history:
        st.info("No market regime data available. Connect to a data source to begin.")
        return
    
    # Get current regime
    current_regime = state.market_regime_history[-1]['regime']
    regime_color = REGIME_COLORS.get(current_regime, "#9E9E9E")
    
    # Display current regime
    st.markdown(f"<h2 style='color: {regime_color};'>Current Regime: {current_regime.title()}</h2>", unsafe_allow_html=True)
    
    # Create columns for regime metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Create regime history timeline
        if len(state.market_regime_history) > 1:
            # Convert to DataFrame
            df = pd.DataFrame(list(state.market_regime_history))
            
            # Calculate regime durations
            for i in range(len(df) - 1):
                df.at[i, 'duration'] = (df.at[i+1, 'timestamp'] - df.at[i, 'timestamp']).total_seconds() / 60  # minutes
            
            # For the last regime, duration is time since it started
            df.at[len(df)-1, 'duration'] = (datetime.now() - df.at[len(df)-1, 'timestamp']).total_seconds() / 60
            
            # Create timeline chart
            fig = px.timeline(
                df, 
                x_start="timestamp", 
                x_end=df['timestamp'] + pd.to_timedelta(df['duration'], unit='m'),
                y="regime",
                color="regime",
                color_discrete_map=REGIME_COLORS,
                title="Market Regime Timeline"
            )
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Regime",
                hovermode="closest"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Regime duration metrics
            st.subheader("Regime Duration Statistics")
            
            # Calculate average duration by regime
            regime_stats = df.groupby('regime')['duration'].agg(['mean', 'count']).reset_index()
            regime_stats.columns = ['Regime', 'Avg. Duration (min)', 'Count']
            
            # Format for display
            regime_stats['Avg. Duration (min)'] = regime_stats['Avg. Duration (min)'].round(1)
            
            st.dataframe(regime_stats, use_container_width=True)
    
    with col2:
        # Create portfolio performance by regime
        if state.portfolio_history:
            # Convert to DataFrame
            portfolio_df = pd.DataFrame(list(state.portfolio_history))
            
            # Function to find regime for a given timestamp
            def find_regime(timestamp):
                for i in range(len(state.market_regime_history)-1, -1, -1):
                    if timestamp >= state.market_regime_history[i]['timestamp']:
                        return state.market_regime_history[i]['regime']
                return "unknown"
            
            # Add regime to portfolio data
            portfolio_df['regime'] = portfolio_df['timestamp'].apply(find_regime)
            
            # Calculate returns by regime
            portfolio_df['value_pct_change'] = portfolio_df['value'].pct_change()
            
            regime_returns = portfolio_df.groupby('regime')['value_pct_change'].agg(['mean', 'std', 'count']).reset_index()
            regime_returns.columns = ['Regime', 'Avg. Return', 'Volatility', 'Observations']
            
            # Calculate annualized metrics (assuming minutely data)
            regime_returns['Annualized Return'] = regime_returns['Avg. Return'] * 525600  # Minutes in a year
            regime_returns['Annualized Volatility'] = regime_returns['Volatility'] * np.sqrt(525600)
            regime_returns['Sharpe Ratio'] = regime_returns['Annualized Return'] / regime_returns['Annualized Volatility']
            
            # Format for display
            regime_returns['Avg. Return'] = regime_returns['Avg. Return'].apply(lambda x: f"{x:.4%}")
            regime_returns['Annualized Return'] = regime_returns['Annualized Return'].apply(lambda x: f"{x:.2%}")
            regime_returns['Annualized Volatility'] = regime_returns['Annualized Volatility'].apply(lambda x: f"{x:.2%}")
            regime_returns['Sharpe Ratio'] = regime_returns['Sharpe Ratio'].round(2)
            
            # Display stats
            st.subheader("Performance by Regime")
            st.dataframe(regime_returns, use_container_width=True)
            
            # Create comparison chart
            if len(regime_returns) > 1:
                # Filter out regimes with fewer than 5 observations
                chart_data = regime_returns[regime_returns['Observations'] > 5].copy()
                
                if not chart_data.empty:
                    # Convert percentage strings back to floats for the chart
                    chart_data['Annualized Return'] = chart_data['Annualized Return'].str.rstrip('%').astype(float) / 100
                    
                    # Create a bar chart
                    fig = px.bar(
                        chart_data,
                        x='Regime',
                        y='Annualized Return',
                        color='Regime',
                        color_discrete_map=REGIME_COLORS,
                        title="Annualized Return by Market Regime"
                    )
                    
                    fig.update_layout(
                        xaxis_title="Market Regime",
                        yaxis_title="Annualized Return",
                        yaxis=dict(tickformat=".0%")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

def display_trading_activity():
    """Display recent trades and strategy allocations"""
    st.header("Trading Activity")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Recent Trades", "Strategy Allocations"])
    
    with tab1:
        if not state.trade_history:
            st.info("No trade data available. Connect to a data source to begin.")
        else:
            # Convert trade history to DataFrame
            df = pd.DataFrame(list(state.trade_history))
            
            # Display recent trades
            st.subheader("Recent Trades")
            
            # Format the dataframe for display
            display_df = df[['timestamp', 'symbol', 'action', 'quantity', 'price', 'value']].copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['price'] = display_df['price'].round(2)
            display_df['value'] = display_df['value'].round(2)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Create a chart of trade values over time
            fig = px.scatter(
                df,
                x='timestamp',
                y='value',
                size='value',
                color='action',
                hover_name='symbol',
                title="Trade Activity Over Time",
                color_discrete_map={'BUY': 'green', 'SELL': 'red'}
            )
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Trade Value ($)",
                hovermode="closest"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create a chart of trades by symbol
            trades_by_symbol = df.groupby('symbol')['value'].sum().reset_index()
            
            fig = px.bar(
                trades_by_symbol,
                x='symbol',
                y='value',
                title="Trading Volume by Symbol",
                color='symbol'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if not state.trade_history or 'weights' not in state.trade_history[-1]:
            st.info("No strategy allocation data available. Connect to a data source to begin.")
        else:
            # Get the latest strategy weights
            latest_weights = state.trade_history[-1]['weights']
            
            # Display current strategy allocations
            st.subheader("Current Strategy Allocations")
            
            # Create a pie chart
            fig = px.pie(
                values=list(latest_weights.values()),
                names=list(latest_weights.keys()),
                title="Strategy Weight Allocation"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display weights as a table
            weight_df = pd.DataFrame({
                'Strategy': list(latest_weights.keys()),
                'Weight': list(latest_weights.values())
            })
            
            weight_df['Weight'] = weight_df['Weight'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(weight_df, use_container_width=True)
            
            # If we have multiple weight updates, show the evolution
            if len(state.trade_history) > 1:
                # Extract weights history
                weights_history = []
                
                for trade in state.trade_history:
                    if 'weights' in trade:
                        weights_history.append({
                            'timestamp': trade['timestamp'],
                            **trade['weights']
                        })
                
                if weights_history:
                    weights_df = pd.DataFrame(weights_history)
                    
                    # Plot weights over time
                    fig = px.line(
                        weights_df,
                        x='timestamp',
                        y=weights_df.columns[1:],  # All columns except timestamp
                        title="Strategy Weights Over Time",
                        line_shape='spline'
                    )
                    
                    fig.update_layout(
                        xaxis_title="Time",
                        yaxis_title="Weight",
                        hovermode="x unified",
                        yaxis=dict(tickformat=".0%")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

def display_risk_dashboard():
    """Display risk management metrics and visualizations"""
    st.header("Risk Management Dashboard")
    
    # Create columns for key risk metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Portfolio Risk",
            value=f"{state.risk_metrics['portfolio_risk']:.1f}%",
            delta=f"{np.random.normal(-3, 1):.1f}%",
            delta_color="inverse"
        )
        st.progress(state.risk_metrics['portfolio_risk'] / 100)
        
    with col2:
        st.metric(
            label="Drawdown",
            value=f"{state.risk_metrics['drawdown']:.1f}%",
            delta=f"{np.random.normal(1.3, 0.5):.1f}%",
            delta_color="inverse"
        )
        st.progress(state.risk_metrics['drawdown'] / 100)
        
    with col3:
        st.metric(
            label="VaR (95%)",
            value=f"${state.risk_metrics['var_95']:,.0f}",
            delta=f"${np.random.normal(150, 50):.0f}",
            delta_color="inverse"
        )
        
    # Create tabs for different risk views
    risk_tab1, risk_tab2, risk_tab3, risk_tab4 = st.tabs([
        "Exposure", "Position Limits", "Correlation Risk", "Psychological Factors"
    ])
    
    with risk_tab1:
        st.subheader("Risk Exposure by Asset Class")
        
        # Use exposure data from state if available, otherwise use mock data
        if state.risk_exposure:
            asset_classes = list(state.risk_exposure.keys())
            exposure_values = [state.risk_exposure[asset_class] for asset_class in asset_classes]
            
            # Mock usage values if not provided
            if state.position_limits and all(asset_class in state.position_limits for asset_class in asset_classes):
                usage_values = [
                    (state.risk_exposure[asset_class] / state.position_limits[asset_class]['max_allocation']) * 100 
                    if state.position_limits[asset_class]['max_allocation'] > 0 else 0
                    for asset_class in asset_classes
                ]
            else:
                # Generate mock usage values
                usage_values = [min(100, value * (1 + np.random.uniform(0.2, 0.8))) for value in exposure_values]
        else:
            # Mock data for asset class exposure
            asset_classes = ["Equity", "Futures", "Forex", "Crypto", "Options"]
            exposure_values = [42, 28, 15, 10, 5]
            usage_values = [42, 56, 38, 33, 25]
        
        # Create a horizontal bar chart for exposure
        fig = go.Figure()
        
        # Add exposure bars
        fig.add_trace(go.Bar(
            y=asset_classes,
            x=exposure_values,
            name='Current Allocation (%)',
            orientation='h',
            marker=dict(color='rgba(46, 125, 247, 0.7)')
        ))
        
        # Add limit usage indicators
        fig.add_trace(go.Bar(
            y=asset_classes,
            x=usage_values,
            name='Limit Usage (%)',
            orientation='h',
            marker=dict(color='rgba(239, 85, 59, 0.7)'),
            opacity=0.5
        ))
        
        fig.update_layout(
            title="Exposure vs. Limits by Asset Class",
            xaxis_title="Percentage",
            barmode='overlay',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk heat map
        st.subheader("Risk Heat Map")
        
        # Use heat map data from state if available, otherwise use mock data
        if state.risk_heat_map:
            heat_map_data = state.risk_heat_map
        else:
            heat_map_data = {
                "Portfolio Leverage": 65,
                "Equity Exposure": 70,
                "Futures Exposure": 56,
                "Forex Exposure": 38,
                "VaR Utilization": 48,
                "Concentration Risk": 62,
                "Correlation Risk": 45,
                "Drawdown": 41
            }
        
        # Create heat map with threshold colors
        def get_color(value):
            if value < 50:
                return "green"
            elif value < 75:
                return "orange"
            else:
                return "red"
        
        # Display heat map as a styled dataframe
        heat_df = pd.DataFrame({
            "Risk Factor": list(heat_map_data.keys()),
            "Usage (%)": list(heat_map_data.values())
        })
        
        st.dataframe(
            heat_df,
            column_config={
                "Usage (%)": st.column_config.ProgressColumn(
                    "Usage (%)",
                    format="%d%%",
                    min_value=0,
                    max_value=100
                )
            },
            hide_index=True,
            use_container_width=True
        )
    
    with risk_tab2:
        st.subheader("Position Size Limits")
        
        # Get position data from state if available, otherwise use mock data
        if hasattr(state, 'position_limits_by_symbol') and state.position_limits_by_symbol:
            positions = state.position_limits_by_symbol
        else:
            # Mock data for positions
            positions = {
                "AAPL": {"value": 12500, "limit": 20000, "usage": 62.5},
                "MSFT": {"value": 15000, "limit": 20000, "usage": 75.0},
                "TSLA": {"value": 8000, "limit": 20000, "usage": 40.0},
                "AMZN": {"value": 10000, "limit": 20000, "usage": 50.0},
                "GOOG": {"value": 9000, "limit": 20000, "usage": 45.0}
            }
        
        # Create a table for position limits
        position_df = pd.DataFrame([
            {
                "Symbol": symbol,
                "Position Value": f"${data['value']:,}",
                "Limit": f"${data['limit']:,}",
                "Usage": data['usage']
            }
            for symbol, data in positions.items()
        ])
        
        st.dataframe(
            position_df,
            column_config={
                "Usage": st.column_config.ProgressColumn(
                    "Limit Usage (%)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Get sector allocation data from state if available, otherwise use mock data
        if hasattr(state, 'sector_allocation') and state.sector_allocation:
            sectors = list(state.sector_allocation.keys())
            allocations = list(state.sector_allocation.values())
        else:
            # Mock sector allocation data
            sectors = ["Technology", "Healthcare", "Financials", "Consumer", "Energy", "Utilities"]
            allocations = [35, 25, 15, 12, 8, 5]
        
        # Sector allocation chart
        st.subheader("Sector Allocation")
        
        fig = px.pie(
            names=sectors,
            values=allocations,
            title="Sector Allocation",
            hole=0.4
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with risk_tab3:
        st.subheader("Correlation Matrix")
        
        # Use correlation matrix from state if available, otherwise use mock data
        if state.correlation_matrix is not None:
            corr_df = state.correlation_matrix
            symbols = corr_df.columns
            corr_data = corr_df.values
        else:
            # Mock correlation matrix
            symbols = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOG", "SPY"]
            corr_data = np.array([
                [1.00, 0.72, 0.45, 0.68, 0.70, 0.65],
                [0.72, 1.00, 0.38, 0.65, 0.82, 0.58],
                [0.45, 0.38, 1.00, 0.42, 0.36, 0.40],
                [0.68, 0.65, 0.42, 1.00, 0.75, 0.60],
                [0.70, 0.82, 0.36, 0.75, 1.00, 0.62],
                [0.65, 0.58, 0.40, 0.60, 0.62, 1.00]
            ])
            corr_df = pd.DataFrame(corr_data, columns=symbols, index=symbols)
        
        fig = px.imshow(
            corr_df,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Position Correlation Matrix"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Highly correlated pairs
        st.subheader("Highly Correlated Pairs")
        
        # Use high correlation pairs from state if available
        if state.high_correlation_pairs:
            high_corr_pairs = state.high_correlation_pairs
        else:
            # Find highly correlated pairs (above 0.7)
            high_corr_pairs = []
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    if corr_data[i, j] > 0.7:
                        high_corr_pairs.append({
                            "Symbol 1": symbols[i],
                            "Symbol 2": symbols[j],
                            "Correlation": corr_data[i, j]
                        })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            st.dataframe(high_corr_df, hide_index=True, use_container_width=True)
        else:
            st.info("No highly correlated pairs found.")
    
    with risk_tab4:
        st.subheader("Psychological Risk Factors")
        
        # Use psychological risk data from state if available
        psych_data = {
            "Current Risk Score": state.risk_metrics["psychological_risk_score"],
            "Risk Level": state.risk_metrics["risk_level"],
            "Adjustment Factor": state.risk_metrics["position_size_modifier"],
            "Trading Block": "None"  # Default value
        }
        
        # Create columns for key metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Psychological Risk Score",
                value=f"{psych_data['Current Risk Score']:.1f}%",
                delta=f"{np.random.normal(0, 2):.1f}%",
                delta_color="inverse"
            )
            st.progress(psych_data['Current Risk Score'] / 100)
        
        with col2:
            st.metric(
                label="Position Size Adjustment",
                value=f"{psych_data['Adjustment Factor'] * 100:.0f}%",
                delta=f"{np.random.normal(0, 5):.0f}%",
                delta_color="inverse"
            )
        
        # Pattern detection - use patterns from state if available
        st.subheader("Detected Patterns")
        
        patterns = state.psychological_patterns if state.psychological_patterns else [
            {"pattern": "Overtrading", "confidence": 75, "description": "Trading frequency above historical average"},
            {"pattern": "Loss Aversion", "confidence": 45, "description": "Tendency to exit winning trades early"}
        ]
        
        for pattern in patterns:
            st.markdown(f"**{pattern['pattern']}** ({pattern.get('confidence', 50)}% confidence)")
            st.caption(pattern.get('description', ''))
            st.progress(pattern.get('confidence', 50) / 100)
        
        # Recommendations - use from state if available
        st.subheader("Recommendations")
        
        recommendations = state.risk_recommendations if state.risk_recommendations else [
            "Consider reducing position sizes by 20% during current psychological state",
            "Implement mandatory break after 2 consecutive losses",
            "Review trading plan before next trade to avoid overtrading",
        ]
        
        for rec in recommendations:
            st.info(rec)

def connect_to_risk_manager(data_source):
    """
    Connect to the risk manager to obtain real risk metrics.
    
    In a real implementation, this would create a connection to the risk manager
    and set up regular updates of risk metrics.
    
    Args:
        data_source: Selected data source configuration
    """
    try:
        # For mock data, just use the mock risk updates
        if data_source == "Mock Data":
            logger.info("Using mock risk data")
            # Start a thread to periodically update mock risk metrics
            threading.Thread(
                target=lambda: periodic_risk_updates(mock=True),
                daemon=True
            ).start()
            return
            
        # In a real implementation, we would initialize a connection to the risk manager
        # For example:
        if state.data_manager and hasattr(state.data_manager, 'adapter'):
            # Initialize risk manager with the adapter from the data manager
            risk_config_path = os.path.join(
                os.path.dirname(__file__), 
                '..', 
                'configs', 
                'risk_config.json'
            )
            
            # Try to initialize the risk manager (wrapped in try/except in case the imports don't work)
            try:
                risk_manager = RiskManager(
                    multi_asset_adapter=state.data_manager.adapter,
                    config_path=risk_config_path
                )
                
                # Start a thread to periodically fetch real risk metrics
                threading.Thread(
                    target=lambda: periodic_risk_updates(
                        mock=False, 
                        risk_manager=risk_manager
                    ),
                    daemon=True
                ).start()
                
                logger.info("Connected to risk manager successfully")
            except Exception as e:
                logger.warning(f"Could not initialize real risk manager: {e}. Using mock data.")
                # Fall back to mock data
                threading.Thread(
                    target=lambda: periodic_risk_updates(mock=True),
                    daemon=True
                ).start()
        else:
            logger.warning("Data manager adapter not available. Using mock risk data.")
            # Fall back to mock data
            threading.Thread(
                target=lambda: periodic_risk_updates(mock=True),
                daemon=True
            ).start()
    
    except Exception as e:
        logger.error(f"Error connecting to risk manager: {e}")
        # Fall back to mock data
        threading.Thread(
            target=lambda: periodic_risk_updates(mock=True),
            daemon=True
        ).start()

def periodic_risk_updates(mock=True, risk_manager=None):
    """
    Periodically fetch and update risk metrics from the risk manager.
    
    Args:
        mock: Whether to use mock data or real risk manager
        risk_manager: Optional risk manager instance for real data
    """
    while state.streaming_active:
        try:
            if mock:
                # Use mock data (already implemented in update_mock_risk_metrics)
                # This gets called from on_bar_update already
                pass
            else:
                # Fetch real risk metrics from the risk manager
                risk_report = risk_manager.get_risk_report()
                
                # Extract metrics
                if risk_report:
                    risk_metrics = {
                        "portfolio_risk": risk_report.get("portfolio_metrics", {}).get("total_exposure_pct", 30),
                        "drawdown": risk_report.get("portfolio_metrics", {}).get("current_drawdown_pct", 5),
                        "var_95": risk_report.get("var", {}).get("var_95_dollars", 2000),
                        "max_drawdown": risk_report.get("portfolio_metrics", {}).get("max_drawdown_pct", 15),
                        "leverage": risk_report.get("portfolio_metrics", {}).get("leverage", 1.0)
                    }
                    
                    # Get asset class exposure
                    exposure = risk_report.get("portfolio_metrics", {}).get("exposure_by_class", {})
                    
                    # Get position limits
                    limits = risk_report.get("limits", {})
                    
                    # Get correlation matrix if available
                    correlation = risk_report.get("portfolio_metrics", {}).get("correlation_matrix")
                    
                    # Get psychological risk data if available
                    psych_data = risk_report.get("psychological", {})
                    
                    # Add to data queue as a risk update
                    state.data_queue.put({
                        'type': 'risk_update',
                        'metrics': risk_metrics,
                        'exposure': exposure,
                        'limits': limits,
                        'correlation': correlation,
                        'psychological': psych_data,
                        'timestamp': datetime.now()
                    })
                    
                    logger.debug("Updated risk metrics from risk manager")
                
            # Sleep for a few seconds before the next update
            time.sleep(5)
                
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
            time.sleep(5)  # Wait before retrying

def initialize_ai_assistant(data_source):
    """
    Initialize the AI assistant for the dashboard.
    
    Args:
        data_source: Selected data source
    """
    try:
        # Initialize assistant with dashboard interface
        if state.data_manager:
            state.ai_assistant = BenBotAssistant(
                data_manager=state.data_manager,
                dashboard_interface=state.data_manager.get_dashboard_interface() if hasattr(state.data_manager, 'get_dashboard_interface') else None
            )
        else:
            # Use mock data
            state.ai_assistant = BenBotAssistant()
        
        # Initialize chat history if empty
        if not state.chat_history:
            # Add welcome message
            state.chat_history.append({
                "role": "assistant",
                "content": "Hello! I'm your trading assistant. I can help you analyze market data, understand trading strategies, and provide recommendations. What would you like to know?",
                "timestamp": datetime.now()
            })
        
        # Generate initial strategy recommendations
        if not state.strategy_recommendations:
            state.strategy_recommendations = generate_mock_strategy_recommendations()
        
        logger.info("AI assistant initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing AI assistant: {e}")
        # Create a mock assistant for demonstration
        state.ai_assistant = None
        
        # Add error message to chat history
        if not state.chat_history:
            state.chat_history.append({
                "role": "assistant",
                "content": "Hello! I'm your trading assistant. I'm running in demo mode currently. How can I help you today?",
                "timestamp": datetime.now()
            })

def generate_mock_strategy_recommendations():
    """Generate mock strategy recommendations for demo purposes"""
    current_time = datetime.now()
    
    return [
        {
            "strategy": "MA Crossover",
            "symbol": "AAPL",
            "action": "BUY",
            "confidence": 0.85,
            "reasoning": "Golden cross pattern formed with 50-day MA crossing above 200-day MA. Volume confirms trend.",
            "timestamp": current_time - timedelta(minutes=30)
        },
        {
            "strategy": "RSI Reversal",
            "symbol": "MSFT",
            "action": "SELL",
            "confidence": 0.76,
            "reasoning": "RSI in overbought territory (78) with bearish divergence on price. Consider taking profits.",
            "timestamp": current_time - timedelta(hours=2)
        },
        {
            "strategy": "Volatility Breakout",
            "symbol": "SPY",
            "action": "BUY",
            "confidence": 0.82,
            "reasoning": "Breaking out of 20-day consolidation range with increasing volume. ATR expanding.",
            "timestamp": current_time - timedelta(hours=4)
        },
        {
            "strategy": "Trend Following",
            "symbol": "QQQ",
            "action": "HOLD",
            "confidence": 0.91,
            "reasoning": "Strong uptrend intact with price above all major moving averages. Continue holding positions.",
            "timestamp": current_time - timedelta(hours=6)
        },
        {
            "strategy": "Sentiment Analysis",
            "symbol": "TSLA",
            "action": "NEUTRAL",
            "confidence": 0.65,
            "reasoning": "Mixed signals in social media sentiment. High volatility expected around earnings. Consider reducing position size.",
            "timestamp": current_time - timedelta(hours=12)
        }
    ]

def process_ai_request(user_message):
    """
    Process a user message with the AI assistant.
    
    Args:
        user_message: User's message string
        
    Returns:
        Assistant's response
    """
    if not user_message.strip():
        return "Please enter a message to continue the conversation."
    
    # Add user message to chat history
    state.chat_history.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.now()
    })
    
    try:
        # Process with real assistant if available
        if state.ai_assistant:
            response = state.ai_assistant.process_message(user_message, context="dashboard")
        else:
            # Generate mock response if assistant not available
            response = generate_mock_assistant_response(user_message)
        
        # Add assistant response to chat history
        state.chat_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now()
        })
        
        return response
    
    except Exception as e:
        error_msg = f"Error processing your request: {str(e)}"
        logger.error(error_msg)
        
        # Add error response to chat history
        state.chat_history.append({
            "role": "assistant",
            "content": f"I'm sorry, I encountered an error processing your request. Please try again or reformulate your question.",
            "timestamp": datetime.now()
        })
        
        return error_msg

def generate_mock_assistant_response(user_message):
    """
    Generate a mock response for the AI assistant in demo mode.
    
    Args:
        user_message: User's message
        
    Returns:
        Mock response string
    """
    # Simple pattern matching for demo responses
    user_message = user_message.lower()
    
    if "hello" in user_message or "hi" in user_message:
        return "Hello! How can I help you with your trading today?"
    
    elif "strategy" in user_message or "recommend" in user_message:
        return "Based on current market conditions, trend-following strategies are performing well for large-cap tech stocks. Consider using moving average crossovers with confirmation from RSI."
    
    elif "market" in user_message:
        return "The market is currently in a cautiously bullish state. The S&P 500 is up 0.7% today with technology and healthcare sectors leading gains."
    
    elif "portfolio" in user_message:
        return "Your portfolio is currently well-diversified with a moderate risk profile. You might consider increasing your allocation to defensive sectors given the current economic uncertainty."
    
    elif "risk" in user_message:
        return "Your current portfolio risk is moderate (32%). The largest risk factors are concentration in technology stocks and potential sensitivity to interest rate changes."
    
    elif "performance" in user_message:
        return "Your portfolio has returned 12.4% year-to-date, outperforming the S&P 500 by 3.2%. Your best performing position is AAPL (+28%) and your worst is XLE (-8%)."
    
    # Default response
    return "I understand you're asking about " + user_message[:20] + "... To provide specific insights on this topic, I would need to analyze your portfolio data in more detail. Can you clarify what aspect you're most interested in?"

def display_ai_assistant():
    """Display AI assistant interface for natural language interaction"""
    st.header("AI Trading Assistant")
    
    # Create tabs for different AI features
    ai_tab1, ai_tab2 = st.tabs(["Chat", "Strategy Recommendations"])
    
    with ai_tab1:
        st.subheader("Trading Chat Assistant")
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for message in state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"**You**: {message['content']}")
                else:
                    st.markdown(f"**Assistant**: {message['content']}")
        
        # Chat input
        user_input = st.text_input("Ask me anything about trading, strategies, or your portfolio:", key="chat_input")
        
        # Process message when user submits
        if st.button("Send", key="send_btn"):
            if user_input:
                # Process in the background to avoid blocking
                with st.spinner("Thinking..."):
                    response = process_ai_request(user_input)
                
                # Clear input after sending (using a trick with session state)
                st.session_state["chat_input"] = ""
                
                # Force rerun to show the new message
                st.experimental_rerun()
        
        # Support instructions
        with st.expander("What can I ask?"):
            st.markdown("""
            You can ask me about:
            - Trading strategies and their performance
            - Market analysis and insights
            - Portfolio performance and optimization
            - Risk management recommendations
            - Technical and fundamental analysis
            - Trading patterns and signals
            """)
    
    with ai_tab2:
        st.subheader("AI Strategy Recommendations")
        
        # Display strategy recommendations
        if state.strategy_recommendations:
            for rec in state.strategy_recommendations:
                # Create colored box based on action
                action = rec.get("action", "NEUTRAL")
                confidence = rec.get("confidence", 0.5) * 100
                
                # Select color based on action
                if action == "BUY":
                    box_color = "rgba(0, 128, 0, 0.2)"
                    border_color = "rgba(0, 128, 0, 0.7)"
                elif action == "SELL":
                    box_color = "rgba(255, 0, 0, 0.2)"
                    border_color = "rgba(255, 0, 0, 0.7)"
                else:
                    box_color = "rgba(128, 128, 128, 0.2)"
                    border_color = "rgba(128, 128, 128, 0.7)"
                
                # Format timestamp
                timestamp = rec.get("timestamp", datetime.now()).strftime("%Y-%m-%d %H:%M")
                
                # Create styled container for recommendation
                st.markdown(f"""
                <div style="background-color: {box_color}; padding: 15px; border-radius: 5px; border: 1px solid {border_color}; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between;">
                        <div><strong>{rec.get("symbol", "Unknown")}</strong> - {rec.get("strategy", "Strategy")}</div>
                        <div><strong>{action}</strong> ({confidence:.1f}% confidence)</div>
                    </div>
                    <p style="margin-top: 10px; margin-bottom: 5px;">{rec.get("reasoning", "")}</p>
                    <div style="font-size: 0.8em; color: gray; text-align: right;">{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No strategy recommendations available at this time.")
        
        # Refresh button
        if st.button("Refresh Recommendations"):
            # In a real implementation, this would fetch new recommendations
            # For demo, just regenerate mock recommendations
            state.strategy_recommendations = generate_mock_strategy_recommendations()
            st.success("Recommendations refreshed!")
            time.sleep(1)
            st.experimental_rerun()

def create_dashboard():
    """Main function to create and display the dashboard"""
    # Set up sidebar and dashboard configuration 
    symbols, update_frequency, time_range, data_source = initialize_dashboard()
    
    # Create tabs for different dashboard sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Portfolio", "Market Data", "Trading Activity", "Market Regime", "Risk Management", "AI Assistant"
    ])
    
    with tab1:
        display_portfolio_performance()
    
    with tab2:
        display_market_data()
    
    with tab3:
        display_trading_activity()
    
    with tab4:
        display_market_regime()
        
    with tab5:
        display_risk_dashboard()
        
    with tab6:
        display_ai_assistant()
    
    # Check for updates in the data queue
    check_data_updates()

if __name__ == "__main__":
    create_dashboard() 