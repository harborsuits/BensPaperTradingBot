#!/usr/bin/env python
"""
Dashboard Launcher Script

This script launches the trading dashboard with all components
for monitoring the trading system during paper trading.
"""
import os
import sys
import time
import subprocess
import argparse
import signal
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dashboard_launcher')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_environment():
    """Load environment variables from .env.paper file."""
    env_file = project_root / '.env.paper'
    
    if not env_file.exists():
        logger.warning(f".env.paper file not found at {env_file}")
        logger.info("Environment variables must be set manually or by running setup_paper_trading_env.sh")
        return False
    
    logger.info(f"Loading environment variables from {env_file}")
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            key, value = line.split('=', 1)
            os.environ[key] = value
    
    # Verify required variables
    required_vars = [
        'TRADING_MODE',
        'TRADING_ALPACA_API_KEY',
        'TRADING_ALPACA_API_SECRET',
        'TRADING_LOG_LEVEL'
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        return False
    
    logger.info("Environment variables loaded successfully")
    return True


def launch_streamlit(dashboard_module, port=None):
    """
    Launch a Streamlit dashboard.
    
    Args:
        dashboard_module: Path to the Streamlit dashboard module
        port: Optional specific port to use
    
    Returns:
        The subprocess object
    """
    cmd = ['streamlit', 'run', str(dashboard_module)]
    
    if port:
        cmd.extend(['--server.port', str(port)])
    
    logger.info(f"Launching Streamlit dashboard: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def main():
    """Main entry point for the dashboard launcher."""
    parser = argparse.ArgumentParser(description="Launch Trading Dashboard")
    parser.add_argument('--port', type=int, help='Port for the main dashboard')
    parser.add_argument('--all', action='store_true', help='Launch all dashboard pages')
    parser.add_argument('--performance', action='store_true', help='Launch performance dashboard')
    parser.add_argument('--system', action='store_true', help='Launch system monitoring dashboard')
    
    args = parser.parse_args()
    
    # Load environment variables
    if not setup_environment():
        logger.error("Failed to setup environment")
        sys.exit(1)
    
    # Determine port from arguments or environment
    port = args.port or os.getenv('TRADING_DASHBOARD_PORT', 8501)
    
    # Prepare dashboard paths
    dashboard_dir = project_root / 'trading_bot' / 'dashboard'
    
    system_dashboard = dashboard_dir / 'pages' / 'system_monitoring.py'
    performance_dashboard = dashboard_dir / 'pages' / 'performance_dashboard.py'
    
    # Track launched processes
    processes = []
    
    try:
        # Determine which dashboards to launch
        if args.all or (not args.performance and not args.system):
            # Launch main dashboard
            main_dashboard = dashboard_dir / 'main.py'
            if not main_dashboard.exists():
                logger.warning(f"Main dashboard not found at {main_dashboard}")
                logger.info("Creating simple default main dashboard")
                
                # Create a simple main dashboard
                with open(main_dashboard, 'w') as f:
                    f.write("""
import streamlit as st

st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("BensBot Trading Dashboard")
st.write("Welcome to the trading system dashboard. Navigate using the sidebar.")

st.sidebar.title("Navigation")
st.sidebar.info(
    "Select a dashboard page from the dropdown menu in the sidebar."
)

st.markdown('''
## Dashboard Pages

### System Monitoring
Monitor system health, data quality, and emergency controls.

### Performance Analytics
Track trading performance, P&L, and strategy metrics.

## Trading Status

The system is currently running in paper trading mode.

''')

# Display trading mode status
trading_mode = "PAPER"  # This would be retrieved from config
if trading_mode == "PAPER":
    st.success("🧪 System is running in PAPER trading mode")
elif trading_mode == "LIVE":
    st.error("💵 System is running in LIVE trading mode")
else:
    st.info("📊 System is in SIMULATION mode")
"""
                    )
            
            processes.append(launch_streamlit(main_dashboard, port=port))
            logger.info(f"Main dashboard running on port {port}")
            
            # Wait for main dashboard to start
            time.sleep(2)
        
        # Launch specific dashboards if requested
        if args.performance:
            if not performance_dashboard.exists():
                logger.error(f"Performance dashboard not found at {performance_dashboard}")
            else:
                perf_port = int(port) + 1
                processes.append(launch_streamlit(performance_dashboard, port=perf_port))
                logger.info(f"Performance dashboard running on port {perf_port}")
        
        if args.system:
            if not system_dashboard.exists():
                logger.error(f"System dashboard not found at {system_dashboard}")
            else:
                sys_port = int(port) + 2
                processes.append(launch_streamlit(system_dashboard, port=sys_port))
                logger.info(f"System dashboard running on port {sys_port}")
        
        # Keep the script running until Ctrl+C
        print("\nDashboard(s) running. Press Ctrl+C to stop.\n")
        
        # Wait for the processes
        for p in processes:
            p.wait()
            
    except KeyboardInterrupt:
        print("\nShutting down dashboards...")
        for p in processes:
            p.terminate()
        
        # Give processes time to terminate gracefully
        time.sleep(2)
        
        # Force kill any remaining processes
        for p in processes:
            if p.poll() is None:
                p.kill()
        
        print("All dashboards stopped.")


if __name__ == "__main__":
    main()
