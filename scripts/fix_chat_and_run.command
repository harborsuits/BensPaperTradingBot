#!/bin/bash

# Change to script directory
cd "$(dirname "$0")"

echo "===================================="
echo "BenBot Chat Assistant Fixer"
echo "===================================="
echo ""
echo "This script will set up the environment and run the dashboard server."
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required dependencies
echo "Installing required dependencies..."
pip install pandas numpy python-dotenv

# Load environment variables
if [ -f ".env" ]; then
  echo "Loading environment variables from .env file..."
  source <(grep -v '^#' .env | sed -E 's/(.+)=(.+)/export \1=\2/g')
  
  # Check for API keys
  if [ -n "$OPENAI_API_KEY" ] || [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$MISTRAL_API_KEY" ] || 
     [ -n "$COHERE_API_KEY" ] || [ -n "$GEMINI_API_KEY" ]; then
    echo "API keys found in .env file - chat will use AI responses"
  else
    echo ""
    echo "⚠️  No API keys found in .env file"
    echo "The chat will use rule-based responses instead of AI"
    echo "Edit the .env file and uncomment/add your API keys to use AI chat"
    echo ""
  fi
else
  echo "⚠️ No .env file found - chat will use rule-based responses"
fi

# Let user know what's happening
echo ""
echo "Starting the dashboard server..."
echo "The chat will be available at http://localhost:8082"
echo "If that port is in use, check the console for the actual port"
echo "Press Ctrl+C to stop the server when done."
echo ""

# Run the dashboard server
python run_dashboard.py 