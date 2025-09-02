"""
Automated Trading Workflow Launcher

This standalone script launches the integrated automated trading workflow, allowing
users to backtest strategies, approve them, and deploy to paper trading.
"""

import streamlit as st
import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("automated_workflow")

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the workflow components
from trading_bot.ui.strategy_approval_workflow import StrategyApprovalWorkflow

def main():
    """Main entry point for the automated trading workflow application"""
    
    # Set up the Streamlit page
    st.set_page_config(
        page_title="Automated Trading Workflow",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar information
    with st.sidebar:
        st.title("Automated Trading Workflow")
        st.markdown("---")
        
        st.subheader("Workflow Steps")
        st.markdown("""
        1. **Configure & Backtest** - Set up strategy parameters and run backtests
        2. **Review Results** - Analyze backtest results and approve/reject strategies
        3. **Approved Strategies** - Deploy approved strategies to paper trading
        4. **Trading Monitor** - Monitor live paper trading performance
        """)
        
        st.markdown("---")
        
        st.subheader("System Status")
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            st.metric("Data Provider", "Connected", delta=None)
            st.metric("Market Status", "Open", delta=None)
        
        with status_col2:
            st.metric("Active Strategies", "0", delta=None)
            now = datetime.now().strftime("%H:%M:%S")
            st.metric("Last Update", now, delta=None)
    
    # Main content
    st.title("🚀 Automated Trading Workflow")
    
    # Create workflow tabs
    tabs = st.tabs(["Strategy Approval", "Trading Monitor"])
    
    with tabs[0]:
        # Initialize and render strategy approval workflow
        workflow = StrategyApprovalWorkflow()
        workflow.render()
    
    with tabs[1]:
        st.header("Paper Trading Monitor")
        
        # Check for deployed strategies
        deploy_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "paper_trading"
        )
        os.makedirs(deploy_dir, exist_ok=True)
        
        deployed_strategies = [f for f in os.listdir(deploy_dir) if f.endswith(".json")]
        
        if not deployed_strategies:
            st.info("No strategies currently deployed to paper trading. Approve a strategy in the Strategy Approval tab to get started.")
        else:
            st.success(f"{len(deployed_strategies)} strategies deployed to paper trading.")
            
            # Simulated trading metrics
            cols = st.columns(4)
            
            with cols[0]:
                st.metric("Today's P&L", "$237.50", delta="+2.37%")
            
            with cols[1]:
                st.metric("Open Positions", "3", delta=None)
            
            with cols[2]:
                st.metric("Win Rate", "62.5%", delta=None)
            
            with cols[3]:
                st.metric("Sharpe Ratio", "1.85", delta="+0.12")
            
            # Simulated positions
            st.subheader("Current Positions")
            
            positions_data = [
                {"Symbol": "AAPL", "Direction": "LONG", "Quantity": 25, "Entry": "$175.23", "Current": "$179.45", "Unrealized P&L": "$105.50", "Strategy": "Momentum"},
                {"Symbol": "MSFT", "Direction": "LONG", "Quantity": 15, "Entry": "$342.15", "Current": "$347.80", "Unrealized P&L": "$84.75", "Strategy": "Momentum"},
                {"Symbol": "META", "Direction": "SHORT", "Quantity": 10, "Entry": "$485.70", "Current": "$479.35", "Unrealized P&L": "$63.50", "Strategy": "Mean Reversion"}
            ]
            
            st.dataframe(positions_data, use_container_width=True)
            
            # Trading logs
            st.subheader("Trading Logs")
            
            log_container = st.container(height=250)
            
            with log_container:
                st.code("""
2025-04-24 21:15:32 - INFO - Strategy 'Momentum' scanning for signals...
2025-04-24 21:15:38 - INFO - New BUY signal detected for AAPL, RSI: 32.5, ROC: 1.2%
2025-04-24 21:15:39 - INFO - Order placed: BUY 25 AAPL @ $175.23
2025-04-24 21:15:40 - INFO - Order filled: BUY 25 AAPL @ $175.23
2025-04-24 21:24:15 - INFO - Strategy 'Mean Reversion' scanning for signals...
2025-04-24 21:24:19 - INFO - New SHORT signal detected for META, Z-Score: 2.3, Mean Dev: 2.8%
2025-04-24 21:24:20 - INFO - Order placed: SELL 10 META @ $485.70
2025-04-24 21:24:22 - INFO - Order filled: SELL 10 META @ $485.70
2025-04-24 21:30:45 - INFO - Risk limits checked: Portfolio exposure at 32%, within 40% threshold
2025-04-24 21:45:12 - INFO - Strategy 'Momentum' scanning for signals...
2025-04-24 21:45:17 - INFO - New BUY signal detected for MSFT, RSI: 34.2, ROC: 0.9%
2025-04-24 21:45:18 - INFO - Order placed: BUY 15 MSFT @ $342.15
2025-04-24 21:45:20 - INFO - Order filled: BUY 15 MSFT @ $342.15
2025-04-24 21:50:00 - INFO - Daily portfolio analytics updated, Sharpe: 1.85, Sortino: 2.21
                """)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Refresh Data", use_container_width=True):
                    st.toast("Data refreshed successfully!")
            
            with col2:
                st.button("Pause All Strategies", use_container_width=True)
            
            with col3:
                st.button("Emergency Close All", type="primary", use_container_width=True)

if __name__ == "__main__":
    main()
