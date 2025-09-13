#!/usr/bin/env python3
"""
Direct Dashboard Launcher for BensBot Trading
This script handles:
1. Adding the proper Python paths for module resolution
2. Checking and installing dependencies 
3. Creating mock modules for any missing components
4. Starting Streamlit directly without Docker
"""
import os
import sys
import subprocess
import importlib
import time
from types import ModuleType

REQUIRED_PACKAGES = [
    "streamlit",
    "pandas",
    "numpy",
    "plotly",
    "matplotlib",
    "websocket-client",
    "yfinance",
    "ta",
    "scikit-learn",
    "pytz",
    "ccxt",
    "requests",
    "psutil"
]

def check_install_dependencies():
    """Check and install missing dependencies."""
    missing_packages = []
    
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✅ {package} already installed")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n🔍 Installing {len(missing_packages)} missing packages...")
        packages_str = " ".join(missing_packages)
        cmd = f"pip3 install --break-system-packages {packages_str}"
        
        try:
            result = subprocess.run(cmd, shell=True, check=True, 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✅ Successfully installed: {packages_str}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install packages: {e.stderr.decode()}")
            return False
    
    return True

def setup_python_paths():
    """Set up Python paths correctly."""
    # Get the absolute path to the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Dashboard directory
    dashboard_dir = current_dir
    # trading_bot directory
    trading_bot_dir = os.path.dirname(current_dir)
    # Project root
    project_root = os.path.dirname(trading_bot_dir)
    
    # Add all paths
    paths_to_add = [
        project_root,
        trading_bot_dir,
        dashboard_dir,
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.append(path)
            print(f"✅ Added to Python path: {path}")
    
    # Set an environment variable to help any subprocess
    os.environ["PYTHONPATH"] = ":".join(paths_to_add)
    
    return project_root, trading_bot_dir, dashboard_dir

def create_mock_modules():
    """Create full mock module structure for any missing imports to make the dashboard work."""
    try:
        # Try importing critical modules including strategy_factory
        import trading_bot.strategies.strategy_factory
        print("✅ Trading bot modules found!")
        return
    except ImportError:
        print("⚠️ Creating comprehensive mock modules for missing trading bot components...")
        
        # Extended module hierarchy with all required modules
        modules_to_mock = [
            'trading_bot',
            'trading_bot.strategies',
            'trading_bot.strategies.base',
            'trading_bot.strategies.forex',
            'trading_bot.strategies.forex.base',
            'trading_bot.strategies.forex.momentum',
            'trading_bot.strategies.strategy_factory', # Missing module
            'trading_bot.orchestration',
            'trading_bot.orchestration.main_orchestrator',
            'trading_bot.data',
            'trading_bot.data.data_manager',
            'trading_bot.data.real_time_provider',
            'trading_bot.core'
        ]
        
        # Create all modules in hierarchy
        for module_name in modules_to_mock:
            if module_name not in sys.modules:
                module = ModuleType(module_name)
                sys.modules[module_name] = module
                
                # Add __path__ attribute to make it recognized as a package
                if '.' in module_name:
                    module.__path__ = []
                
                print(f"📦 Created mock module: {module_name}")
        
        # Create mock StrategyFactory class
        class MockStrategyFactory:
            @staticmethod
            def create_strategy(*args, **kwargs):
                return None
                
            @staticmethod
            def register_strategy(*args, **kwargs):
                pass
                
        # Add the mock class to the strategy_factory module
        sys.modules['trading_bot.strategies.strategy_factory'].StrategyFactory = MockStrategyFactory
        
        # Create other necessary mock classes
        sys.modules['trading_bot.orchestration.main_orchestrator'].MainOrchestrator = type('MainOrchestrator', (), {})
        sys.modules['trading_bot.data.data_manager'].DataManager = type('DataManager', (), {})
        sys.modules['trading_bot.data.real_time_provider'].RealTimeProvider = type('RealTimeProvider', (), {})
        
        print("✅ Mock trading system components created successfully")

def start_streamlit(dashboard_dir):
    """Start the Streamlit server."""
    app_path = os.path.join(dashboard_dir, 'app.py')
    
    print("\n🚀 Starting Dashboard...")
    print(f"📊 Dashboard will be available at: http://localhost:8501")
    print("👉 Press Ctrl+C to stop the dashboard\n")
    
    cmd = ["streamlit", "run", app_path, "--server.port=8501"]
    
    try:
        process = subprocess.Popen(
            cmd,
            env=os.environ,
            cwd=dashboard_dir
        )
        return process
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return None

def open_browser():
    """Open the browser to the dashboard."""
    print("🌐 Opening browser...")
    time.sleep(3)  # Wait for Streamlit to start
    
    try:
        import webbrowser
        webbrowser.open("http://localhost:8501")
    except Exception as e:
        print(f"❌ Could not automatically open browser: {e}")
        print("Please manually navigate to http://localhost:8501")

def main():
    """Main launcher function."""
    print("\n" + "=" * 60)
    print("      🤖 BensBot Trading Dashboard - Direct Launch 🚀")
    print("=" * 60 + "\n")
    
    # 1. Setup Python paths
    project_root, trading_bot_dir, dashboard_dir = setup_python_paths()
    
    # 2. Check and install dependencies
    if not check_install_dependencies():
        print("❌ Failed to install required dependencies. Exiting.")
        sys.exit(1)
    
    # 3. Create mock modules if needed
    create_mock_modules()
    
    # 4. Start Streamlit
    process = start_streamlit(dashboard_dir)
    if not process:
        print("❌ Failed to start the dashboard. Exiting.")
        sys.exit(1)
    
    # 5. Open browser
    open_browser()
    
    # Wait for process to complete or Ctrl+C
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n⏹️ Stopping dashboard...")
        process.terminate()
        print("✅ Dashboard stopped.")

if __name__ == "__main__":
    main()
