# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to run the BenBot API with real AI integration
"""

import os
import sys
import subprocess
import platform
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BenBot Launcher")

# Project path
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_PATH)

def check_packages():
    """Install required packages if they're missing"""
    required_packages = {
        'openai': 'openai>=1.0.0,<2.0.0',
        'anthropic': 'anthropic>=0.5.0',
        'pyyaml': 'pyyaml>=6.0',
        'fastapi': 'fastapi>=0.95.0',
        'uvicorn': 'uvicorn>=0.21.0'
    }
    
    missing_packages = []
    
    for pkg, version in required_packages.items():
        try:
            __import__(pkg)
            logger.info(f"✓ {pkg} is installed")
        except ImportError:
            missing_packages.append(version)
    
    if missing_packages:
        logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
        try:
            # Use pip directly for better compatibility
            cmd = [sys.executable, "-m", "pip", "install", "--user"] + missing_packages
            subprocess.check_call(cmd)
            logger.info("✓ All required packages installed successfully")
        except Exception as e:
            logger.error(f"Failed to install packages: {e}")
            logger.info("Please manually install the following packages:")
            for pkg in missing_packages:
                logger.info(f"  pip install {pkg}")

def ensure_config_files():
    """Ensure AI configuration files exist"""
    config_dir = os.path.join(PROJECT_PATH, "trading_bot", "config")
    os.makedirs(config_dir, exist_ok=True)
    
    # Create AI config if it doesn't exist
    ai_config_path = os.path.join(config_dir, "ai_config.yaml")
    if not os.path.exists(ai_config_path):
        with open(ai_config_path, 'w') as f:
            f.write("""ai_assistant:
  provider: openai
  model: gpt-4-turbo
  system_prompt: "You are BenBot, an AI assistant for a trading bot system. You help users analyze market conditions, review trading strategies, and manage their portfolio. Be concise, accurate, and focus on providing actionable trading insights."
""")
        logger.info(f"Created AI configuration file: {ai_config_path}")
    
    # Create AI keys file if it doesn't exist
    ai_keys_path = os.path.join(config_dir, "ai_keys.yaml")
    if not os.path.exists(ai_keys_path):
        with open(ai_keys_path, 'w') as f:
            f.write("""# API Keys for AI services
openai: sk-your-openai-api-key
claude: sk-your-claude-api-key
""")
        logger.info(f"Created AI keys file: {ai_keys_path}")
        logger.info("⚠️ Please edit the ai_keys.yaml file with your actual API keys")

def run_api_server():
    """Run the API server with the correct Python path"""
    logger.info("Starting BenBot API server with real AI integration...")
    logger.info(f"Project path: {PROJECT_PATH}")
    
    # Make sure the Python path is set correctly
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_PATH
    
    try:
        # Import API modules to test if they're available
        from trading_bot.api import app
        logger.info("✓ API modules imported successfully")
        
        # Start the server
        logger.info("Starting FastAPI server...")
        import uvicorn
        uvicorn.run("trading_bot.api.app:app", host="0.0.0.0", port=5000, reload=True)
        
    except ImportError as e:
        logger.error(f"Failed to import API modules: {e}")
        logger.info("Trying to run server with system command...")
        
        # Run as subprocess if import fails
        cmd = [sys.executable, "-m", "uvicorn", "trading_bot.api.app:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]
        subprocess.run(cmd, env=env)

if __name__ == "__main__":
    print("="*50)
    print("BenBot AI Trading Assistant Launcher")
    print("="*50)
    
    # Check and install required packages
    check_packages()
    
    # Ensure config files exist
    ensure_config_files()
    
    # Run the API server
    run_api_server()
