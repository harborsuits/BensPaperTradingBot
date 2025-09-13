#!/usr/bin/env python3
"""
Trade Journal System Initialization Script

This utility script helps initialize the trade journal system with a provided template.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import Trade Journal System
try:
    from analytics.trade_journal_system import TradeJournalSystem
except ImportError:
    # Adjust import path if running from different directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from trading_bot.analytics.trade_journal_system import TradeJournalSystem

def load_json_data(file_path):
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON data from {file_path}: {str(e)}")
        raise

def initialize_journal(template_file, journal_dir):
    """
    Initialize the Trade Journal System with a provided template.
    
    Args:
        template_file: Path to template JSON file
        journal_dir: Directory for journal data
        
    Returns:
        bool: True if successful
    """
    try:
        # Load template data
        logger.info(f"Loading template from {template_file}")
        template_data = load_json_data(template_file)
        
        # Create journal system
        logger.info(f"Initializing journal system in {journal_dir}")
        journal = TradeJournalSystem(journal_dir=journal_dir)
        
        # Save template
        templates_dir = os.path.join(journal_dir, "templates")
        os.makedirs(templates_dir, exist_ok=True)
        
        template_path = os.path.join(templates_dir, "default_template.json")
        with open(template_path, 'w') as f:
            json.dump(template_data.get("journal_template", {}), f, indent=2)
        
        logger.info(f"Saved journal template to {template_path}")
        
        # Create initial config
        config_path = os.path.join(journal_dir, "config.json")
        config = {
            "schema_version": template_data.get("schema_version", "2.1.0"),
            "last_updated": datetime.now().isoformat(),
            "journal_template": template_data.get("journal_template", {})
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created journal configuration at {config_path}")
        
        # Test creating a sample trade
        sample_trade = {
            "trade_metadata": {
                "trade_id": "SAMPLE-TRADE",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "timestamp": datetime.now().isoformat(),
                "ticker": "AAPL",
                "asset_class": "equity",
                "position_type": "long"
            },
            "execution_details": {
                "entry": {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "time": datetime.now().strftime("%H:%M %p ET"),
                    "price": 180.50,
                    "quantity": 10
                }
            }
        }
        
        trade_id = journal.start_new_trade(sample_trade)
        logger.info(f"Created sample trade with ID: {trade_id}")
        
        # Add sample exit data
        exit_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M %p ET"),
            "price": 185.75,
            "quantity": 10,
            "exit_reason": "target_reached",
            "exit_condition": "Sample exit condition"
        }
        
        journal.close_trade(trade_id, exit_data)
        logger.info(f"Closed sample trade {trade_id}")
        
        logger.info(f"Trade Journal System successfully initialized in {journal_dir}")
        logger.info(f"To integrate with your trading bot, use the JournalIntegration class from analytics.journal_integration")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing journal: {str(e)}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Initialize Trade Journal System")
    parser.add_argument("--template", "-t", default="trade_journal_template.json", 
                        help="Path to template JSON file (default: trade_journal_template.json)")
    parser.add_argument("--dir", "-d", default="journal", 
                        help="Directory for journal data (default: journal)")
    
    args = parser.parse_args()
    
    # Check if template file exists
    if not os.path.exists(args.template):
        logger.error(f"Template file not found: {args.template}")
        sys.exit(1)
    
    # Initialize journal
    success = initialize_journal(args.template, args.dir)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 