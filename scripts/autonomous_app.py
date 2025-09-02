"""
Autonomous Trading App

This standalone script provides a focused interface for the autonomous trading system,
allowing users to automatically scan markets, generate strategies, backtest them,
and approve them for paper trading.
"""

import streamlit as st
import sys
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("autonomous_app")

# Set page configuration
st.set_page_config(
    page_title="Autonomous Trading System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the autonomous UI
try:
    from trading_bot.autonomous import AutonomousUI
    autonomous_ui_available = True
except ImportError as e:
    autonomous_ui_available = False
    logger.error(f"Failed to import autonomous UI: {e}")

def main():
    """Main entry point for the autonomous trading app"""
    
    # Create sidebar
    with st.sidebar:
        st.title("Autonomous Trading System")
        st.markdown("---")
        
        st.markdown("""
        ### System Components
        
        1. **Market Scanner** - Automatically identifies trading opportunities
        2. **Strategy Generator** - Creates optimized strategies
        3. **Backtesting Engine** - Tests strategies on historical data
        4. **Approval Interface** - Review and approve strategies
        5. **Paper Trading** - Deploy approved strategies
        """)
        
        st.markdown("---")
        
        # System status
        st.subheader("System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Data Provider", "Connected", delta=None)
            st.metric("Market Status", "Open", delta=None)
        
        with col2:
            st.metric("Running Strategies", "3", delta=None)
            now = datetime.now().strftime("%H:%M:%S")
            st.metric("Last Update", now, delta=None)
        
        # Links
        st.markdown("---")
        st.markdown("[Return to Main Dashboard](http://localhost:8501)")
    
    # Main content
    st.title("🤖 Autonomous Trading System")
    
    if autonomous_ui_available:
        # Create and render the autonomous UI
        autonomous_ui = AutonomousUI()
        autonomous_ui.render()
    else:
        st.error("Autonomous trading components not available. Please check your installation.")
        
        # Display fallback message
        st.markdown("""
        ### Autonomous Trading System
        
        This system would automatically:
        
        1. Scan the market for trading opportunities
        2. Generate optimized trading strategies
        3. Backtest strategies on historical data
        4. Present only strategies that meet performance criteria
        5. Allow one-click approval and deployment to paper trading
        
        Please install the required components to activate this functionality.
        """)

if __name__ == "__main__":
    main()
