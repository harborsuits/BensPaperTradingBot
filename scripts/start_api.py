#!/usr/bin/env python3
"""
Script to start the trading bot API server with the BenBot AI Assistant.
Sets the proper Python path and handles initialization.
"""
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import FastAPI app
from trading_bot.api.app import app

# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    print("Starting trading bot API server with BenBot Assistant...")
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
