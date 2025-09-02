#!/usr/bin/env python3
"""
Debug script to test the integration between BenBot Assistant and the Trading Orchestrator
"""

import os
import sys
import logging
import traceback
import json
from pprint import pprint

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("debug_benbot_orchestrator")

def main():
    """Test the integration between BenBot and the orchestrator"""
    
    logger.info("Starting debug of BenBot Assistant with orchestrator integration")
    
    # First check if trading_bot is available
    try:
        logger.info("Attempting to import components...")
        
        # Import the BenBotAssistant class
        from trading_bot.assistant.benbot_assistant import BenBotAssistant
        
        # Import other components we need for testing
        try:
            from trading_bot.orchestration.main_orchestrator import MainOrchestrator
            from trading_bot.data.data_manager import DataManager
            orchestrator = MainOrchestrator()
            data_manager = DataManager()
            logger.info("✅ Successfully imported MainOrchestrator and DataManager")
        except Exception as e:
            logger.error(f"❌ Failed to import orchestrator components: {e}")
            logger.info("Creating mock orchestrator for testing")
            orchestrator = type('MockOrchestrator', (), {
                'run_pipeline': lambda *args, **kwargs: "Mock pipeline executed",
                'get_approved_opportunities': lambda: [{"symbol": "AAPL", "strategy": "Test"}],
                'get_market_regime': lambda: {"regime": "Bullish", "confidence": 0.85}
            })()
            data_manager = None
    except Exception as e:
        logger.error(f"❌ Failed to import required components: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Initialize required directories if they don't exist
    for dirname in ["data", "results", "models"]:
        os.makedirs(dirname, exist_ok=True)
    
    # Create trading context
    trading_context = {
        "orchestrator": orchestrator,
        "data_manager": data_manager,
    }
    
    # Initialize the BenBot Assistant
    try:
        logger.info("Initializing BenBot Assistant...")
        assistant = BenBotAssistant(
            data_manager=data_manager,
            dashboard_interface=None,
            data_dir="data",
            results_dir="results",
            models_dir="models"
        )
        
        # Store trading context
        assistant.trading_context = trading_context
        logger.info("✅ BenBot Assistant initialized with trading context")
    except Exception as e:
        logger.error(f"❌ Failed to initialize BenBot Assistant: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Test trading queries
    logger.info("\n==== Testing Trading Queries ====")
    
    # Test queries
    test_queries = [
        "Run the trading pipeline",
        "Show me approved opportunities",
        "What's the current market regime?",
        "Explain the current market conditions",
        "What trading opportunities are available?"
    ]
    
    for query in test_queries:
        logger.info(f"\n🔍 Testing query: \"{query}\"")
        try:
            response = assistant.handle_query(query)
            logger.info(f"✅ Response: {response}")
        except Exception as e:
            logger.error(f"❌ Error on query \"{query}\": {e}")
            logger.error(traceback.format_exc())
    
    logger.info("\nDebug complete. Check the logs above for any issues.")

if __name__ == "__main__":
    main() 