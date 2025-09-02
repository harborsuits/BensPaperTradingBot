#!/usr/bin/env python3
"""
Example script to update the sector rotation framework via the webhook API.

This script loads the sector rotation framework data from a JSON file 
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

def update_sector_rotation_framework(api_url, framework_data):
    """
    Update the sector rotation framework via the API.
    
    Args:
        api_url: API URL for the update endpoint
        framework_data: Sector rotation framework data
        
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
    parser = argparse.ArgumentParser(description="Update sector rotation framework")
    parser.add_argument("--file", "-f", required=True, help="Path to JSON file with framework data")
    parser.add_argument("--url", "-u", default="http://localhost:5000/macro-guidance/sector-rotation-update", 
                        help="API endpoint URL (default: http://localhost:5000/macro-guidance/sector-rotation-update)")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)
    
    try:
        # Load framework data
        logger.info(f"Loading sector rotation framework from {args.file}")
        framework_data = load_json_data(args.file)
        
        # Send data to API
        logger.info(f"Sending framework data to {args.url}")
        response = update_sector_rotation_framework(args.url, framework_data)
        
        # Process response
        if response.get("status") == "success":
            logger.info("Successfully updated sector rotation framework")
            logger.info(f"Updated {response.get('phases_updated')} cycle phases: {', '.join(response.get('phases', []))}")
            logger.info(f"Framework version: {response.get('framework_version')}")
        else:
            logger.error(f"Failed to update framework: {response.get('message')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error updating sector rotation framework: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 