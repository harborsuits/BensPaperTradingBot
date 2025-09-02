#!/usr/bin/env python3
"""
BenBot AI Setup Script
This script implements all necessary fixes to get the AI integration working properly.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BenBot")

# Project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
VENV_PATH = Path.home() / "benbot_env"
CONFIG_DIR = PROJECT_ROOT / "trading_bot" / "config"

def create_venv():
    """Create a virtual environment"""
    if VENV_PATH.exists():
        logger.info(f"Virtual environment already exists at {VENV_PATH}")
        return True
    
    logger.info(f"Creating virtual environment at {VENV_PATH}")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VENV_PATH)], check=True)
        logger.info("Virtual environment created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create virtual environment: {e}")
        return False

def fix_assistant():
    """Fix the BenBot Assistant code"""
    assistant_file = PROJECT_ROOT / "trading_bot" / "assistant" / "benbot_assistant.py"
    if not assistant_file.exists():
        logger.error(f"Assistant file not found: {assistant_file}")
        return False
    
    logger.info("Fixing BenBot Assistant...")
    
    # Read current file
    with open(assistant_file, "r") as f:
        content = f.readlines()
    
    # Create backup if it doesn't exist
    backup_file = assistant_file.with_suffix(".py.bak")
    if not backup_file.exists():
        with open(backup_file, "w") as f:
            f.writelines(content)
        logger.info(f"Backup created at {backup_file}")
    
    # Fix imports and error handling in process_query
    new_process_query = """    def process_query(self, query: str):
        \"\"\"
        Process a natural language query and return a response.
        
        Args:
            query: The user's natural language query
            
        Returns:
            Response string with results or action confirmation
        \"\"\"
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
            logger.error(f"Error using AI service: {e}")
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
    
    # Find process_query method and replace it
    in_process_query = False
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(content):
        if "def process_query" in line:
            in_process_query = True
            start_idx = i
        elif in_process_query and line.startswith("    def "):
            end_idx = i
            break
    
    if start_idx is not None and end_idx is not None:
        # Replace the method
        content = content[:start_idx] + new_process_query.splitlines(True) + content[end_idx:]
        
        # Write the updated file
        with open(assistant_file, "w") as f:
            f.writelines(content)
        
        logger.info("Successfully fixed BenBot Assistant")
        return True
    else:
        logger.error("Could not find process_query method to replace")
        return False

def ensure_config_files():
    """Ensure AI configuration files exist"""
    # Create config directory
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Create ai_config.yaml
    ai_config_file = CONFIG_DIR / "ai_config.yaml"
    if not ai_config_file.exists():
        with open(ai_config_file, "w") as f:
            f.write("""ai_assistant:
  provider: openai
  model: gpt-4-turbo
  system_prompt: "You are BenBot, an AI assistant for a trading bot system. You help users analyze market conditions, review trading strategies, and manage their portfolio. Be concise, accurate, and focus on providing actionable trading insights."
""")
        logger.info(f"Created AI config file: {ai_config_file}")
    
    # Create ai_keys.yaml
    ai_keys_file = CONFIG_DIR / "ai_keys.yaml"
    if not ai_keys_file.exists():
        with open(ai_keys_file, "w") as f:
            f.write("""openai: sk-your-openai-api-key
claude: sk-your-claude-api-key
""")
        logger.info(f"Created AI keys file: {ai_keys_file}")
        logger.info("Note: You'll need to edit ai_keys.yaml to add your actual API keys.")
    
    return True

def create_startup_script():
    """Create a script to start the API server"""
    script_file = PROJECT_ROOT / "start_ai_api.sh"
    
    script_content = f"""#!/bin/bash
# Start the BenBot API with AI integration
GREEN='\\033[0;32m'
BLUE='\\033[0;34m'
RED='\\033[0;31m'
NC='\\033[0m'

echo -e "${{BLUE}}Starting BenBot API with AI integration...{{NC}}"

# Paths
VENV_PATH="{VENV_PATH}"
PROJECT_PATH="{PROJECT_ROOT}"

# Activate virtual environment
if [ -f "${{VENV_PATH}}/bin/activate" ]; then
    source "${{VENV_PATH}}/bin/activate"
    echo -e "${{GREEN}}Virtual environment activated{{NC}}"
else
    echo -e "${{RED}}Virtual environment not found at ${{VENV_PATH}}{{NC}}"
    echo -e "${{BLUE}}Continuing without virtual environment{{NC}}"
fi

# Set Python path
export PYTHONPATH="${{PROJECT_PATH}}"

echo -e "${{GREEN}}Starting API server...{{NC}}"
echo -e "${{BLUE}}API will be available at: http://localhost:5000{{NC}}"
echo -e "${{BLUE}}Press Ctrl+C to stop{{NC}}"

cd "${{PROJECT_PATH}}"
python -m trading_bot.api.app
"""
    
    with open(script_file, "w") as f:
        f.write(script_content)
    
    os.chmod(script_file, 0o755)
    logger.info(f"Created startup script: {script_file}")
    return True

def create_dashboard_launcher():
    """Create a launcher for the dashboard with AI integration"""
    script_file = PROJECT_ROOT / "launch_ai_dashboard.sh"
    
    script_content = f"""#!/bin/bash
# Launch the trading dashboard with AI integration
GREEN='\\033[0;32m'
BLUE='\\033[0;34m'
YELLOW='\\033[1;33m'
NC='\\033[0m'

echo -e "${{BLUE}}=================================={{NC}}"
echo -e "${{GREEN}}BenBot AI Dashboard Launcher{{NC}}"
echo -e "${{BLUE}}=================================={{NC}}"

# Start API server in a new terminal window
echo -e "${{BLUE}}Starting API server in new terminal...{{NC}}"
osascript -e 'tell application "Terminal" to do script "cd {PROJECT_ROOT} && ./start_ai_api.sh"'

# Give API server time to start
echo -e "${{YELLOW}}Waiting for API server to start...{{NC}}"
sleep 3

# Open browser to dashboard
echo -e "${{GREEN}}Opening dashboard in browser...{{NC}}"
open "http://localhost:3000"

echo -e "${{GREEN}}Setup complete!{{NC}}"
echo -e "${{BLUE}}Your dashboard is available at: http://localhost:3000{{NC}}"
echo -e "${{BLUE}}The API server is running in a separate terminal window.{{NC}}"
echo -e "${{BLUE}}=================================={{NC}}"
"""
    
    with open(script_file, "w") as f:
        f.write(script_content)
    
    os.chmod(script_file, 0o755)
    logger.info(f"Created dashboard launcher: {script_file}")
    return True

def update_api_client():
    """Update API client for better fallback handling"""
    api_file = PROJECT_ROOT / "new-trading-dashboard" / "src" / "services" / "api.ts"
    if not api_file.exists():
        logger.error(f"API client not found: {api_file}")
        return False
    
    logger.info("Updating API client for better fallback...")
    
    # Read current file
    with open(api_file, "r") as f:
        content = f.read()
    
    # Create backup if doesn't exist
    backup_file = api_file.with_suffix(".ts.bak")
    if not backup_file.exists():
        with open(backup_file, "w") as f:
            f.write(content)
        logger.info(f"Backup created at {backup_file}")
    
    # Update retry interval
    updated = content.replace(
        "const CONNECTION_RETRY_INTERVAL = 30000;", 
        "const CONNECTION_RETRY_INTERVAL = 5000; // 5 seconds between reconnection attempts"
    )
    
    # Update shouldTryRealApi function
    old_function = """const shouldTryRealApi = () => {
  // If we've never tried or if it's been a while since last failed attempt
  return isBackendAvailable || 
         lastConnectionAttempt === 0 || 
         (Date.now() - lastConnectionAttempt) > CONNECTION_RETRY_INTERVAL;
};"""
    
    new_function = """const shouldTryRealApi = () => {
  // More aggressive retry approach
  const shouldTry = isBackendAvailable || 
         lastConnectionAttempt === 0 || 
         (Date.now() - lastConnectionAttempt) > CONNECTION_RETRY_INTERVAL;
  
  // Log connection attempts for debugging
  if (shouldTry) {
    console.log('Attempting to connect to real AI backend at: ' + API_BASE_URL);
  }
  
  return shouldTry;
};"""
    
    updated = updated.replace(old_function, new_function)
    
    # Write updated content
    with open(api_file, "w") as f:
        f.write(updated)
    
    logger.info("Successfully updated API client")
    return True

def install_requirements():
    """Install required packages"""
    logger.info("Installing required packages...")
    
    requirements = [
        "openai>=1.0.0",
        "anthropic>=0.5.0",
        "pyyaml>=6.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0"
    ]
    
    # Try to install packages
    pip_path = VENV_PATH / "bin" / "pip3"
    if pip_path.exists():
        try:
            for pkg in requirements:
                logger.info(f"Installing {pkg}...")
                subprocess.run([str(pip_path), "install", pkg], check=True)
            logger.info("All packages installed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to install packages: {e}")
    else:
        logger.error(f"Pip not found at {pip_path}")
    
    # If we get here, something failed, provide manual instructions
    logger.warning("Could not automatically install packages")
    logger.info("Please manually install these packages in your environment:")
    for pkg in requirements:
        logger.info(f"  pip install {pkg}")
    
    return False

def main():
    """Main function to run all fixes"""
    logger.info("Starting BenBot AI integration setup")
    
    # Create virtual environment
    create_venv()
    
    # Fix BenBot Assistant
    fix_assistant()
    
    # Ensure config files
    ensure_config_files()
    
    # Create startup scripts
    create_startup_script()
    create_dashboard_launcher()
    
    # Update API client
    update_api_client()
    
    # Try to install requirements
    install_requirements()
    
    logger.info("Setup completed!")
    logger.info("\nTo use your AI-powered trading bot:")
    logger.info("1. Run ./launch_ai_dashboard.sh to start everything")
    logger.info("2. Or run ./start_ai_api.sh to just start the API server")
    logger.info("3. Access your dashboard at http://localhost:3000")

if __name__ == "__main__":
    main()
