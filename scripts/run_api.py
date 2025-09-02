#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run script for the Trading Bot API
"""

import uvicorn
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    # Start the API server
    uvicorn.run(
        "trading_bot.api.app_with_engine:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True
    )
