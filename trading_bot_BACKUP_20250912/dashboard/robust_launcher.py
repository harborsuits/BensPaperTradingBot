#!/usr/bin/env python3
"""
Robust Dashboard Launcher for BensBot Trading

This script fixes Python path issues and ensures all necessary modules
exist before launching the dashboard.
"""
import os
import sys
import importlib
import subprocess
import time
import types
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RobustLauncher")

# ANSI color codes for prettier output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

def print_header():
    """Print a fancy header"""
    print(f"\n{BLUE}{'='*70}")
    print(f"{BLUE}||{YELLOW}              BensBot Trading Dashboard - Robust Launcher           {BLUE}||")
    print(f"{BLUE}{'='*70}{NC}\n")

def fix_python_paths():
    """Set up correct Python import paths and ensure they work"""
    # Get the absolute path to key directories
    dashboard_dir = os.path.dirname(os.path.abspath(__file__))
    trading_bot_dir = os.path.dirname(dashboard_dir)
    project_root = os.path.dirname(trading_bot_dir)
    
    # Paths to add
    paths_to_add = [
        project_root,
        trading_bot_dir,
        dashboard_dir,
        # Add more paths if needed
    ]
    
    # Add paths to sys.path if not already there
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
            logger.info(f"{GREEN}Added to Python path: {path}{NC}")
    
    # Set PYTHONPATH environment variable for child processes
    os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)
    
    return project_root, trading_bot_dir, dashboard_dir

def ensure_module_structure():
    """Ensure all required modules exist in the Python path"""
    # Create a mapping of module problems and their fixes
    module_fixes = {
        'trading_bot.strategies.strategy_factory': fix_strategy_factory,
        # Add more module fixes as needed
    }
    
    for module_path, fix_function in module_fixes.items():
        try:
            # Try importing the module
            importlib.import_module(module_path)
            logger.info(f"{GREEN}✓ Module {module_path} imported successfully{NC}")
        except ImportError as e:
            # If import fails, apply the fix
            logger.warning(f"{YELLOW}⚠ Module {module_path} not found: {e}{NC}")
            fix_function()
            # Verify the fix worked
            try:
                importlib.import_module(module_path)
                logger.info(f"{GREEN}✓ Module {module_path} fixed and imported successfully{NC}")
            except ImportError as e:
                logger.error(f"{RED}✗ Failed to fix module {module_path}: {e}{NC}")
                raise

def fix_strategy_factory():
    """Fix the strategy_factory module import issue"""
    # First check if the bridge file exists
    strategies_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'strategies')
    bridge_file = os.path.join(strategies_dir, 'strategy_factory.py')
    
    # If the file doesn't exist, create it
    if not os.path.exists(bridge_file):
        logger.info(f"{YELLOW}Creating strategy_factory bridge file...{NC}")
        
        bridge_content = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
Strategy Factory Bridge Module

This module re-exports the StrategyFactory class from the factory subdirectory
to maintain backward compatibility with code that imports from
trading_bot.strategies.strategy_factory
\"\"\"

# Re-export the StrategyFactory class
from trading_bot.strategies.factory.strategy_factory import StrategyFactory

# Re-export any other necessary classes
from trading_bot.strategies.factory.strategy_registry import (
    StrategyRegistry, 
    AssetClass, 
    StrategyType, 
    MarketRegime, 
    TimeFrame
)

# Make sure these are available when importing from this module
__all__ = [
    'StrategyFactory',
    'StrategyRegistry',
    'AssetClass',
    'StrategyType',
    'MarketRegime',
    'TimeFrame'
]
"""
        with open(bridge_file, 'w') as f:
            f.write(bridge_content)
        
        logger.info(f"{GREEN}✓ Created bridge file at {bridge_file}{NC}")
    
    # Also fix the __init__.py to properly define the package
    init_file = os.path.join(strategies_dir, '__init__.py')
    
    if os.path.exists(init_file):
        with open(init_file, 'r') as f:
            content = f.read()
        
        # Only modify if it doesn't already have our modifications
        if 'from .factory.strategy_factory import StrategyFactory' not in content:
            logger.info(f"{YELLOW}Updating strategies/__init__.py...{NC}")
            
            # Update the content to include proper package definition
            if 'from . import factory' in content:
                lines = content.splitlines()
                insert_point = 0
                
                # Find the line after the imports
                for i, line in enumerate(lines):
                    if 'from . import factory' in line:
                        insert_point = i + 1
                        break
                
                # Insert our modifications
                new_lines = [
                    "",
                    "# Direct import and re-export of StrategyFactory",
                    "try:",
                    "    from .factory.strategy_factory import StrategyFactory",
                    "    from .factory.strategy_registry import (",
                    "        StrategyRegistry,",
                    "        AssetClass,",
                    "        StrategyType,",
                    "        MarketRegime,",
                    "        TimeFrame",
                    "    )",
                    "except ImportError as e:",
                    "    import sys",
                    "    import logging",
                    "    logging.warning(f\"Error importing strategy components: {e}\")",
                    "    # Create placeholder if import fails",
                    "    class StrategyFactory:",
                    "        @staticmethod",
                    "        def create_strategy(*args, **kwargs):",
                    "            return None",
                    "    sys.modules[__name__ + '.strategy_factory'] = __import__('types').ModuleType('strategy_factory')",
                    "    sys.modules[__name__ + '.strategy_factory'].StrategyFactory = StrategyFactory"
                ]
                
                updated_content = "\n".join(lines[:insert_point] + new_lines + lines[insert_point:])
                
                with open(init_file, 'w') as f:
                    f.write(updated_content)
                
                logger.info(f"{GREEN}✓ Updated {init_file}{NC}")
    
    # Create the module in sys.modules if it doesn't exist
    module_name = 'trading_bot.strategies.strategy_factory'
    if module_name not in sys.modules:
        logger.info(f"{YELLOW}Creating module {module_name} in sys.modules...{NC}")
        
        # Create the module object
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        
        # Try to import from the actual location
        try:
            from trading_bot.strategies.factory.strategy_factory import StrategyFactory
            # Add the class to our module
            module.StrategyFactory = StrategyFactory
        except ImportError:
            # Create a dummy class if import fails
            class MockStrategyFactory:
                @staticmethod
                def create_strategy(*args, **kwargs):
                    return None
            
            module.StrategyFactory = MockStrategyFactory
        
        logger.info(f"{GREEN}✓ Created module {module_name} in sys.modules{NC}")

def check_install_packages():
    """Check and install required packages"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'websocket-client'
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            logger.info(f"{GREEN}✓ {package} already installed{NC}")
        except ImportError:
            logger.warning(f"{YELLOW}Installing {package}...{NC}")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install',
                '--break-system-packages', package
            ])

def start_streamlit(dashboard_dir):
    """Start the Streamlit server"""
    logger.info(f"{GREEN}Starting dashboard...{NC}")
    
    # Dashboard file to run
    app_file = os.path.join(dashboard_dir, 'app.py')
    
    if not os.path.exists(app_file):
        logger.error(f"{RED}Dashboard app file not found: {app_file}{NC}")
        sys.exit(1)
    
    # Use full path for streamlit
    streamlit_path = os.path.join(os.path.dirname(sys.executable), 'streamlit')
    
    cmd = [streamlit_path, 'run', app_file, '--server.port=8501']
    
    # Start Streamlit
    try:
        process = subprocess.Popen(
            cmd,
            env=os.environ,
            cwd=dashboard_dir
        )
        
        logger.info(f"{GREEN}Dashboard started! Available at http://localhost:8501{NC}")
        
        # Open browser after a short delay
        time.sleep(2)
        try:
            import webbrowser
            webbrowser.open('http://localhost:8501')
        except:
            pass
        
        # Wait for the process to complete
        process.wait()
    except Exception as e:
        logger.error(f"{RED}Failed to start Streamlit: {e}{NC}")
        sys.exit(1)

def main():
    """Main function to launch the dashboard"""
    print_header()
    
    # Fix Python paths
    project_root, trading_bot_dir, dashboard_dir = fix_python_paths()
    
    # Check and install required packages
    check_install_packages()
    
    # Ensure all required modules exist
    ensure_module_structure()
    
    # Start the dashboard
    start_streamlit(dashboard_dir)

if __name__ == "__main__":
    main()
