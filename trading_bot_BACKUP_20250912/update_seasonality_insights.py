#!/usr/bin/env python3
"""
Example script to update the seasonality insights framework via the webhook API.

This script loads the seasonality insights framework data from a JSON file 
and sends it to the trading bot's webhook API to update the framework.
"""

import json
import argparse
import requests
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_json_data(file_path):
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON data from {file_path}: {str(e)}")
        raise

def update_seasonality_framework(api_url, framework_data):
    """
    Update the seasonality insights framework via the API.
    
    Args:
        api_url: API URL for the update endpoint
        framework_data: Seasonality insights framework data
        
    Returns:
        API response as a dictionary
    """
    try:
        response = requests.post(
            api_url,
            json=framework_data,
            headers={"Content-Type": "application/json"}
        )
        
        # Check for successful response
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred: {str(e)}")
        if response:
            logger.error(f"Response: {response.text}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending request: {str(e)}")
        raise

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Update seasonality insights framework")
    parser.add_argument("--file", "-f", required=True, help="Path to JSON file with framework data")
    parser.add_argument("--url", "-u", default="http://localhost:5000/macro-guidance/seasonality-update", 
                        help="API endpoint URL (default: http://localhost:5000/macro-guidance/seasonality-update)")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)
    
    try:
        # Load framework data
        logger.info(f"Loading seasonality insights framework from {args.file}")
        framework_data = load_json_data(args.file)
        
        # Check for the required structure
        if "seasonality_insights" not in framework_data:
            logger.error("The framework data must contain a 'seasonality_insights' key at the top level")
            sys.exit(1)
        
        # Send data to API
        logger.info(f"Sending framework data to {args.url}")
        response = update_seasonality_framework(args.url, framework_data)
        
        # Process response
        if response.get("status") == "success":
            logger.info("Successfully updated seasonality insights framework")
            details = response.get("details", {})
            if "months_updated" in details:
                logger.info(f"Updated {details.get('months_updated')} months: {', '.join(details.get('months', []))}")
            if "recurring_patterns_updated" in details:
                logger.info(f"Updated {details.get('recurring_patterns_updated')} recurring patterns: {', '.join(details.get('recurring_patterns', []))}")
            logger.info(f"Framework version: {details.get('framework_version')}")
        else:
            logger.error(f"Failed to update framework: {response.get('message')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error updating seasonality insights framework: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 