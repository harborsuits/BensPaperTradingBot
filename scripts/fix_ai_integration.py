#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix AI Integration Script for BenBot
This script implements all necessary fixes to get the AI integration working properly.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BenBot-Fixer")

# Set paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
VENV_PATH = Path.home() / "benbot_env"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements_ai.txt"
CONFIG_DIR = PROJECT_ROOT / "trading_bot" / "config"

# Create requirements file if it doesn't exist
def create_requirements_file():
    if not REQUIREMENTS_FILE.exists():
        with open(REQUIREMENTS_FILE, "w") as f:
            f.write("""fastapi==0.103.1
uvicorn==0.23.2
openai==1.0.0
anthropic==0.5.0
pyyaml==6.0.1
requests==2.31.0
""")
        logger.info(f"Created requirements file at {REQUIREMENTS_FILE}")

# Setup virtual environment
def setup_virtual_env():
    logger.info(f"Setting up virtual environment at {VENV_PATH}")
    
    # Create virtual environment
    if not VENV_PATH.exists():
        try:
            subprocess.run([sys.executable, "-m", "venv", str(VENV_PATH)], 
                          check=True, capture_output=True)
            logger.info(f"Created virtual environment at {VENV_PATH}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            logger.error(f"Output: {e.stdout.decode() if e.stdout else ''}")
            logger.error(f"Error: {e.stderr.decode() if e.stderr else ''}")
            return False
    else:
        logger.info(f"Virtual environment already exists at {VENV_PATH}")
    
    return True

# Install requirements
def install_requirements():
    logger.info("Installing required packages...")
    
    # Determine pip path
    if os.name == 'nt':  # Windows
        pip_path = VENV_PATH / "Scripts" / "pip"
    else:  # Unix/Mac
        pip_path = VENV_PATH / "bin" / "pip"
    
    try:
        subprocess.run([str(pip_path), "install", "-r", str(REQUIREMENTS_FILE)],
                      check=True, capture_output=True)
        logger.info("Successfully installed required packages")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install packages: {e}")
        logger.error(f"Output: {e.stdout.decode() if e.stdout else ''}")
        logger.error(f"Error: {e.stderr.decode() if e.stderr else ''}")
        return False

# Fix BenBotAssistant for better dependency handling
def fix_benbot_assistant():
    file_path = PROJECT_ROOT / "trading_bot" / "assistant" / "benbot_assistant.py"
    if not file_path.exists():
        logger.error(f"BenBot Assistant file not found at {file_path}")
        return False
    
    logger.info(f"Fixing BenBot Assistant for better dependency handling")
    
    # Read the file
    with open(file_path, "r") as f:
        content = f.read()
    
    # Backup the file
    backup_path = file_path.with_suffix(".py.bak")
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        logger.info(f"Backed up original file to {backup_path}")
    
    # Fix imports section
    import_fixes = """
import logging
import re
from typing import Dict, List, Optional, Union, Any
import json
import os
from datetime import datetime

# Setup logger
logger = logging.getLogger(__name__)

# Import NLP and AI libraries if available
try:
    import numpy as np
    NLP_LIBRARIES_AVAILABLE = True
except ImportError:
    NLP_LIBRARIES_AVAILABLE = False
    logger.warning("NLP libraries not available")
    
# Import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Install with: pip install openai")

# Import Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic library not available. Install with: pip install anthropic")
    
# Import YAML
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("YAML library not available. Install with: pip install pyyaml")
"""

    # Replace the imports section
    import_pattern = r"import logging.*?logger\s*=\s*logging\.getLogger\(__name__\)"
    if re.search(import_pattern, content, re.DOTALL):
        content = re.sub(import_pattern, import_fixes, content, flags=re.DOTALL)
    else:
        # If we can't find the exact pattern, inject at the top after docstring
        docstring_end = content.find('"""', content.find('"""') + 3) + 3
        content = content[:docstring_end] + "\n" + import_fixes + content[docstring_end:]
    
    # Fix the process_query method for better error handling
    process_query_fix = """
    def process_query(self, query: str):
        """
        Process a natural language query and return a response.
        
        Args:
            query: The user's natural language query
            
        Returns:
            Response string with results or action confirmation
        """
        # Clean up the input
        query = query.strip()
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        response = None
        try:
            # Try AI service if configured
            if self.ai_service != 'local':
                response = self._get_ai_response(query)
                if response:
                    logger.info("Got response from AI service")
        except Exception as e:
            logger.exception(f"Error using AI service: {e}")
            # Fall back to intent matching
            response = None
        
        # If AI didn't work, use intent matching
        if not response:
            logger.info("Using intent matching for response")
            response = self._match_intent(query)
        
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
"""

    # Replace the process_query method
    if "def process_query" in content:
        pattern = r"def process_query.*?return response\n"
        content = re.sub(pattern, process_query_fix, content, flags=re.DOTALL)
    
    # Write the updated content back
    with open(file_path, "w") as f:
        f.write(content)
    
    logger.info("Successfully updated BenBot Assistant")
    return True

# Ensure AI configuration files exist
def ensure_ai_config_files():
    logger.info("Setting up AI configuration files")
    
    # Create config directory if needed
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Create ai_config.yaml if it doesn't exist
    ai_config_file = CONFIG_DIR / "ai_config.yaml"
    if not ai_config_file.exists():
        with open(ai_config_file, "w") as f:
            f.write("""ai_assistant:
  provider: openai
  model: gpt-4-turbo
  system_prompt: "You are BenBot, an AI assistant for a trading bot system. You help users analyze market conditions, review trading strategies, and manage their portfolio. Be concise, accurate, and focus on providing actionable trading insights."
""")
        logger.info(f"Created AI config file at {ai_config_file}")
    
    # Create ai_keys.yaml if it doesn't exist
    ai_keys_file = CONFIG_DIR / "ai_keys.yaml"
    if not ai_keys_file.exists():
        with open(ai_keys_file, "w") as f:
            f.write("""openai: sk-your-openai-api-key
claude: sk-your-claude-api-key
""")
        logger.info(f"Created AI keys file template at {ai_keys_file}")

# Create startup script for API server
def create_api_startup_script():
    script_path = PROJECT_ROOT / "start_ai_api.sh"
    
    with open(script_path, "w") as f:
        f.write(f"""#!/bin/bash
# Start the BenBot API with AI integration

# Colors
GREEN='\\033[0;32m'
BLUE='\\033[0;34m'
RED='\\033[0;31m'
NC='\\033[0m' # No Color

echo -e "${{BLUE}}Starting BenBot API with AI integration...{{NC}}"

# Set paths
VENV_PATH="{VENV_PATH}"
PROJECT_PATH="{PROJECT_ROOT}"

# Activate virtual environment
if [ -f "${{VENV_PATH}}/bin/activate" ]; then
    echo -e "${{BLUE}}Activating virtual environment...{{NC}}"
    source "${{VENV_PATH}}/bin/activate"
    echo -e "${{GREEN}}Virtual environment activated.{{NC}}"
else
    echo -e "${{RED}}Virtual environment not found at ${{VENV_PATH}}{{NC}}"
    echo -e "${{BLUE}}Continuing without virtual environment...{{NC}}"
fi

# Set Python path
export PYTHONPATH="${{PROJECT_PATH}}"

# Start API server
echo -e "${{GREEN}}Starting API server at http://localhost:5000{{NC}}"
echo -e "${{BLUE}}Press Ctrl+C to stop the server{{NC}}"
echo -e "${{GREEN}}============================================{{NC}}"

cd "${{PROJECT_PATH}}"
python -m trading_bot.api.app
""")
    
    # Make executable
    os.chmod(script_path, 0o755)
    logger.info(f"Created API startup script at {script_path}")

# Create a wrapper script for React dashboard
def create_dashboard_wrapper():
    script_path = PROJECT_ROOT / "launch_dashboard.sh"
    
    with open(script_path, "w") as f:
        f.write(f"""#!/bin/bash
# Launch the React dashboard with instructions

# Colors
GREEN='\\033[0;32m'
BLUE='\\033[0;34m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

echo -e "${{BLUE}}====================================={{NC}}"
echo -e "${{GREEN}}BenBot AI Trading Dashboard Launcher{{NC}}"
echo -e "${{BLUE}}====================================={{NC}}"

# Set project directory
PROJECT_DIR="{PROJECT_ROOT}"
DASHBOARD_DIR="$PROJECT_DIR/new-trading-dashboard"

# Open new terminal to start API server
echo -e "${{GREEN}}Opening new terminal to start AI API server...{{NC}}"
open -a Terminal "${{PROJECT_DIR}}/start_ai_api.sh"

# Wait for API to start
echo -e "${{YELLOW}}Waiting for API to start up...{{NC}}"
sleep 3

# Check if browser is installed
BROWSER="open"
if command -v $BROWSER &> /dev/null; then
    echo -e "${{GREEN}}Opening dashboard in browser...{{NC}}"
    $BROWSER "http://localhost:3000" &
else
    echo -e "${{YELLOW}}Could not open browser automatically.{{NC}}"
    echo -e "${{YELLOW}}Please open http://localhost:3000 in your browser.{{NC}}"
fi

echo -e "${{BLUE}}====================================={{NC}}"
echo -e "${{GREEN}}Setup Complete!{{NC}}"
echo -e "${{BLUE}}The AI API server should be running in a separate terminal.{{NC}}"
echo -e "${{BLUE}}Access your dashboard at: http://localhost:3000{{NC}}"
echo -e "${{BLUE}}====================================={{NC}}"
""")
    
    # Make executable
    os.chmod(script_path, 0o755)
    logger.info(f"Created dashboard launcher script at {script_path}")

# Update API client for better connection handling
def update_api_client():
    file_path = PROJECT_ROOT / "new-trading-dashboard" / "src" / "services" / "api.ts"
    if not file_path.exists():
        logger.error(f"API client file not found at {file_path}")
        return False
    
    logger.info(f"Updating API client for better connection handling")
    
    # Read the file
    with open(file_path, "r") as f:
        content = f.read()
    
    # Backup the file
    backup_path = file_path.with_suffix(".ts.bak")
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        logger.info(f"Backed up original file to {backup_path}")
    
    # Update retry interval
    retry_pattern = r"const CONNECTION_RETRY_INTERVAL = \d+;"
    if re.search(retry_pattern, content):
        content = re.sub(retry_pattern, "const CONNECTION_RETRY_INTERVAL = 5000; // 5 seconds", content)
    
    # Update the shouldTryRealApi function
    should_try_pattern = r"const shouldTryRealApi = \(\) => \{.*?\};"
    should_try_replacement = """const shouldTryRealApi = () => {
  // Always try to connect to the real API more aggressively
  const shouldTry = isBackendAvailable || 
         lastConnectionAttempt === 0 || 
         (Date.now() - lastConnectionAttempt) > CONNECTION_RETRY_INTERVAL;
  
  // Log connection attempts for debugging
  if (shouldTry) {
    console.log('Attempting to connect to real AI backend at: ' + API_BASE_URL);
  }
  
  return shouldTry;
};"""

    if re.search(should_try_pattern, content, re.DOTALL):
        content = re.sub(should_try_pattern, should_try_replacement, content, flags=re.DOTALL)
    
    # Write the updated content back
    with open(file_path, "w") as f:
        f.write(content)
    
    logger.info("Successfully updated API client")
    return True

# Main function to run all fixes
def main():
    logger.info("Starting BenBot AI integration fixes")
    
    # Create requirements file
    create_requirements_file()
    
    # Setup virtual environment
    if not setup_virtual_env():
        logger.error("Failed to set up virtual environment")
        return
    
    # Install requirements
    if not install_requirements():
        logger.warning("Failed to install some requirements")
    
    # Fix BenBot Assistant
    fix_benbot_assistant()
    
    # Ensure AI config files
    ensure_ai_config_files()
    
    # Create startup scripts
    create_api_startup_script()
    create_dashboard_wrapper()
    
    # Update API client
    update_api_client()
    
    logger.info("All fixes completed successfully!")
    logger.info("\nTo start the system:")
    logger.info("1. Run ./launch_dashboard.sh to start everything")
    logger.info("2. Or run './start_ai_api.sh' to start just the backend API")
    logger.info("\nYou can then access your dashboard at: http://localhost:3000")

if __name__ == "__main__":
    main()
