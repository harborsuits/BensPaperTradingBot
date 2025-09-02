"""
Helper module to add the EvoTrader module to the Python path.
"""

import os
import sys

def add_evotrader_to_path():
    """Add the EvoTrader module to the Python path."""
    # Get the absolute path to the root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, "../.."))
    
    # Add the root directory to the Python path if not already there
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        print(f"Added {root_dir} to Python path")
    
    # Verify that evotrader can be imported
    try:
        import evotrader
        print(f"Successfully imported evotrader module from {evotrader.__file__}")
    except ImportError as e:
        print(f"Warning: Failed to import evotrader module: {str(e)}")
        print(f"Current sys.path: {sys.path}")

# Add EvoTrader to path when this module is imported
add_evotrader_to_path()
