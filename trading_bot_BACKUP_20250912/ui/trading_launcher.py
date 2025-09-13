"""
Trading Launcher UI - Simplified interface for starting and monitoring trading
"""

import streamlit as st
import pandas as pd
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import trading components
try:
    from trading_bot.data.market_data_provider import create_data_provider
    from trading_bot.portfolio_state import PortfolioStateManager
    from trading_bot.strategies.momentum import MomentumStrategy
    from trading_bot.strategies.mean_reversion import MeanReversionStrategy
    from trading_bot.strategies.trend_following import TrendFollowingStrategy
    from trading_bot.strategies.volatility_breakout import VolatilityBreakout
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    logging.warning(f"Trading components not available: {e}")

logger = logging.getLogger(__name__)

class TradingLauncher:
    """Simple interface for starting and monitoring trading"""
    
    def __init__(self):
        """Initialize the trading launcher"""
        # Initialize state if not already present
        if "trading_active" not in st.session_state:
            st.session_state.trading_active = False
        if "trading_thread" not in st.session_state:
            st.session_state.trading_thread = None
        if "trading_logs" not in st.session_state:
            st.session_state.trading_logs = []
        if "portfolio" not in st.session_state:
            st.session_state.portfolio = None
        
        # Initialize components if available
        if COMPONENTS_AVAILABLE:
            self.data_provider = create_data_provider("yahoo")  # Default to Yahoo
            self.portfolio_manager = PortfolioStateManager()
            
            # Initialize strategies
            self.strategies = {
                "Momentum": MomentumStrategy(),
                "Mean Reversion": MeanReversionStrategy(),
                "Trend Following": TrendFollowingStrategy(),
                "Volatility Breakout": VolatilityBreakout()
            }
        
    def render(self):
        """Render the trading launcher UI"""
        st.title("🚀 Trading Launcher")
        
        if not COMPONENTS_AVAILABLE:
            st.error("Trading components are not available. Please check your installation.")
            return
        
        # Create two columns
        col1, col2 = st.columns([2, 3])
        
        with col1:
            self._render_trading_controls()
        
        with col2:
            self._render_trading_monitor()
    
    def _render_trading_controls(self):
        """Render trading configuration and control buttons"""
        st.subheader("Trading Configuration")
        
        with st.form("trading_config"):
            # Basic settings
            tickers = st.multiselect(
                "Trading Symbols", 
                ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ"],
                ["AAPL", "MSFT"]
            )
            
            strategy_type = st.selectbox(
                "Strategy", 
                list(self.strategies.keys())
            )
            
            # Trading parameters
            st.subheader("Trading Parameters")
            
            capital = st.number_input("Trading Capital ($)", 
                                      min_value=1000, 
                                      max_value=1000000, 
                                      value=10000,
                                      step=1000)
            
            risk_level = st.slider("Risk Level", 1, 5, 2, 
                                  help="1: Very Conservative, 5: Aggressive")
            
            position_size_pct = st.slider("Max Position Size (%)", 1, 100, 10,
                                         help="Maximum percentage of capital for a single position")
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                use_stoploss = st.checkbox("Use Stop Loss", value=True)
                stoploss_pct = st.slider("Stop Loss (%)", 1, 20, 5)
                
                use_takeprofit = st.checkbox("Use Take Profit", value=True) 
                takeprofit_pct = st.slider("Take Profit (%)", 1, 50, 15)
                
                trading_hours = st.multiselect(
                    "Trading Hours",
                    ["All Day", "Market Open", "First Hour", "Last Hour"],
                    ["Market Open"]
                )
            
            # Submit button
            submitted = st.form_submit_button("Apply Settings")
            
            if submitted:
                st.success("Trading settings applied!")
                
                # Store settings in session state
                st.session_state.trading_settings = {
                    "tickers": tickers,
                    "strategy": strategy_type,
                    "capital": capital,
                    "risk_level": risk_level,
                    "position_size_pct": position_size_pct,
                    "use_stoploss": use_stoploss,
                    "stoploss_pct": stoploss_pct,
                    "use_takeprofit": use_takeprofit,
                    "takeprofit_pct": takeprofit_pct,
                    "trading_hours": trading_hours
                }
        
        # Control buttons
        st.subheader("Trading Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.trading_active:
                if st.button("▶️ Start Trading", key="start_trading"):
                    self._start_trading()
            else:
                st.button("⏸️ Pause Trading", key="pause_trading", disabled=True)
        
        with col2:
            if st.session_state.trading_active:
                if st.button("⏹️ Stop Trading", key="stop_trading"):
                    self._stop_trading()
            else:
                st.button("⏹️ Stop Trading", key="stop_trading", disabled=True)
    
    def _render_trading_monitor(self):
        """Render trading monitoring section"""
        st.subheader("Trading Status")
        
        # Status indicator
        status_col, time_col = st.columns(2)
        
        with status_col:
            if st.session_state.trading_active:
                st.markdown("#### Status: 🟢 ACTIVE")
            else:
                st.markdown("#### Status: ⚪ INACTIVE")
        
        with time_col:
            st.markdown(f"#### Time: {datetime.now().strftime('%H:%M:%S')}")
            st.markdown(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        
        # Portfolio overview
        st.subheader("Portfolio Overview")
        
        if st.session_state.portfolio is not None:
            # Convert portfolio to DataFrame for display
            portfolio_data = st.session_state.portfolio
            
            # Display cash and total value
            cash_col, value_col = st.columns(2)
            with cash_col:
                st.metric("Cash Available", f"${portfolio_data.get('cash', 0):,.2f}")
            with value_col:
                st.metric("Total Value", f"${portfolio_data.get('total_value', 0):,.2f}")
            
            # Display positions
            positions = portfolio_data.get("positions", {})
            if positions:
                positions_df = pd.DataFrame([
                    {
                        "Symbol": symbol,
                        "Quantity": pos.get("quantity", 0),
                        "Avg Price": f"${pos.get('avg_price', 0):.2f}",
                        "Current Value": f"${pos.get('current_value', 0):.2f}",
                        "P&L": f"${pos.get('unrealized_pnl', 0):.2f}",
                        "P&L %": f"{pos.get('unrealized_pnl_pct', 0):.2f}%"
                    }
                    for symbol, pos in positions.items()
                ])
                
                st.dataframe(positions_df)
            else:
                st.info("No open positions")
        else:
            st.info("Portfolio data will appear when trading is active")
        
        # Trading logs
        st.subheader("Trading Logs")
        
        log_container = st.container()
        
        # Display the latest logs (last 10)
        with log_container:
            for log in list(st.session_state.trading_logs)[-10:]:
                st.text(log)
    
    def _start_trading(self):
        """Start the trading process"""
        if not st.session_state.trading_active:
            # Set trading active flag
            st.session_state.trading_active = True
            
            # Log starting
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} - Trading started with {st.session_state.trading_settings['strategy']} strategy"
            st.session_state.trading_logs.append(log_entry)
            
            # Simulate portfolio data for demo
            self._simulate_portfolio()
            
            # In a real implementation, start a trading thread
            # st.session_state.trading_thread = threading.Thread(target=self._trading_loop)
            # st.session_state.trading_thread.daemon = True
            # st.session_state.trading_thread.start()
            
            st.experimental_rerun()
    
    def _stop_trading(self):
        """Stop the trading process"""
        if st.session_state.trading_active:
            # Set trading active flag
            st.session_state.trading_active = False
            
            # Log stopping
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} - Trading stopped"
            st.session_state.trading_logs.append(log_entry)
            
            # In a real implementation, signal the trading thread to stop
            # if st.session_state.trading_thread is not None:
            #     st.session_state.trading_thread = None
            
            st.experimental_rerun()
    
    def _simulate_portfolio(self):
        """Simulate portfolio data for demonstration"""
        # Get settings
        settings = st.session_state.trading_settings
        tickers = settings["tickers"]
        capital = settings["capital"]
        
        # Create a simulated portfolio
        positions = {}
        total_position_value = 0
        
        for ticker in tickers:
            # Simulate a position
            quantity = int(capital * 0.2 / 150)  # Assume average price of $150
            avg_price = 150 + (hash(ticker) % 50)  # Pseudo-random price
            current_price = avg_price * (1 + (hash(ticker) % 20) / 100)  # Slight variation
            current_value = quantity * current_price
            unrealized_pnl = current_value - (quantity * avg_price)
            unrealized_pnl_pct = (unrealized_pnl / (quantity * avg_price)) * 100
            
            positions[ticker] = {
                "quantity": quantity,
                "avg_price": avg_price,
                "current_price": current_price,
                "current_value": current_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct
            }
            
            total_position_value += current_value
        
        # Calculate remaining cash
        cash = capital - total_position_value
        
        # Create portfolio object
        st.session_state.portfolio = {
            "cash": cash,
            "total_value": cash + total_position_value,
            "positions": positions
        }
        
        # Add some log entries
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.trading_logs.append(f"{timestamp} - Portfolio initialized with {len(tickers)} symbols")
        
        # Add sample trade logs
        for ticker in tickers:
            entry_time = (datetime.now() - timedelta(minutes=hash(ticker) % 60)).strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.trading_logs.append(f"{entry_time} - BUY {ticker}: {positions[ticker]['quantity']} shares @ ${positions[ticker]['avg_price']:.2f}")
    
    def _trading_loop(self):
        """
        The main trading loop - would be implemented with real trading logic
        This is a placeholder for demonstration
        """
        pass
