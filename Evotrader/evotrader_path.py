"""
Module path helper for EvoTrader.

This module adds the project root to the Python path,
allowing imports to work properly across the codebase.
"""

import os
import sys

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add to Python path if not already present
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"Added {PROJECT_ROOT} to Python path")

# Verify the import works
try:
    import evotrader
    print(f"Successfully imported evotrader module from {evotrader.__file__}")
except ImportError as e:
    print(f"Warning: Could not import evotrader module: {e}")
