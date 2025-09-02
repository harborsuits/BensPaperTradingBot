#!/usr/bin/env python3
"""
Debug script for BenBot Assistant

This script tests the BenBot Assistant functionality directly without the Streamlit UI.
This helps isolate any issues specific to the assistant vs. UI integration issues.
"""

import os
import sys
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("debug_benbot")

# Add the current directory to Python path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def main():
    """Main debug function"""
    print("=== BenBot Assistant Debug Tool ===")
    
    # Import BenBot Assistant
    try:
        from trading_bot.assistant.benbot_assistant import BenBotAssistant
        print("✅ Successfully imported BenBotAssistant")
    except ImportError as e:
        print(f"❌ Failed to import BenBotAssistant: {e}")
        print("\nPossible solutions:")
        print("1. Make sure trading_bot is installed: pip install -e .")
        print("2. Check that assistant/benbot_assistant.py exists in the trading_bot package")
        return
    
    # Try to import dependencies
    try:
        from trading_bot.data.data_manager import DataManager
        print("✅ Successfully imported DataManager")
    except ImportError as e:
        print(f"❌ Failed to import DataManager: {e}")
    
    try:
        from trading_bot.learning.backtest_learner import BacktestLearner
        print("✅ Successfully imported BacktestLearner")
    except ImportError as e:
        print(f"❌ Failed to import BacktestLearner: {e}")
    
    try:
        from trading_bot.dashboard.backtest_dashboard import BacktestDashboard
        print("✅ Successfully imported BacktestDashboard")
    except ImportError as e:
        print(f"❌ Failed to import BacktestDashboard: {e}")
    
    # Check directories existence
    data_dir = os.path.join(current_dir, "data")
    results_dir = os.path.join(current_dir, "results")
    models_dir = os.path.join(current_dir, "models")
    
    if not os.path.exists(data_dir):
        print(f"⚠️ Data directory not found: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        print(f"  Created directory: {data_dir}")
    else:
        print(f"✅ Data directory exists: {data_dir}")
    
    if not os.path.exists(results_dir):
        print(f"⚠️ Results directory not found: {results_dir}")
        os.makedirs(results_dir, exist_ok=True)
        print(f"  Created directory: {results_dir}")
    else:
        print(f"✅ Results directory exists: {results_dir}")
    
    if not os.path.exists(models_dir):
        print(f"⚠️ Models directory not found: {models_dir}")
        os.makedirs(models_dir, exist_ok=True)
        print(f"  Created directory: {models_dir}")
    else:
        print(f"✅ Models directory exists: {models_dir}")
    
    print("\n=== Initializing BenBot Assistant ===")
    
    # Try initializing the assistant
    try:
        # First try with just directories
        assistant = BenBotAssistant(
            data_dir=data_dir,
            results_dir=results_dir,
            models_dir=models_dir
        )
        print("✅ Successfully initialized BenBot Assistant with directories only")
    except Exception as e:
        print(f"❌ Failed to initialize BenBot Assistant with directories only: {e}")
        print(traceback.format_exc())
        
        # Try a more minimal initialization
        try:
            assistant = BenBotAssistant()
            print("✅ Successfully initialized BenBot Assistant with no parameters")
        except Exception as e:
            print(f"❌ Failed to initialize BenBot Assistant with no parameters: {e}")
            print(traceback.format_exc())
            return
    
    print("\n=== Testing BenBot Assistant Responses ===")
    
    # Test queries to try
    test_queries = [
        "What can you help me with?",
        "How does the trading bot work?",
        "What strategies are available?",
        "Show me recent backtest results"
    ]
    
    # Try each query
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            response = assistant.process_message(query)
            if isinstance(response, dict) and "text" in response:
                response = response["text"]
            print(f"Response: {response}")
            print("✅ Successfully got response")
        except Exception as e:
            print(f"❌ Error getting response: {e}")
            print(traceback.format_exc())
    
    print("\n=== Debug Complete ===")
    print("If all tests passed, BenBot Assistant is working correctly.")
    print("If there were errors, check the traceback for specific issues.")

if __name__ == "__main__":
    main() 