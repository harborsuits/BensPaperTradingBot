#!/usr/bin/env python3
"""
Test script for anomaly detection endpoints.
Tests both REST API and WebSocket connections.
"""

import asyncio
import json
import requests
import websockets
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("anomaly_test")

# API endpoint configuration
API_BASE_URL = "http://localhost:8000"
WS_BASE_URL = "ws://localhost:8000"

async def test_rest_endpoint():
    """Test the REST API endpoint for anomaly detection"""
    try:
        # Test the base endpoint
        url = f"{API_BASE_URL}/api/orchestrator/anomalies"
        logger.info(f"Testing REST endpoint: {url}")
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Successfully received anomaly data:")
            logger.info(f"Timestamp: {data.get('timestamp', 'N/A')}")
            logger.info(f"Anomaly count: {data.get('anomalyCount', 0)}")
            
            # Print details of each anomaly
            for i, anomaly in enumerate(data.get('anomalies', [])):
                logger.info(f"Anomaly #{i+1}: {anomaly.get('description', 'No description')}")
                logger.info(f"  Type: {anomaly.get('type', 'unknown')}")
                logger.info(f"  Severity: {anomaly.get('severity', 0)} ({anomaly.get('severityLabel', 'unknown')})")
                logger.info(f"  Affected assets: {', '.join(anomaly.get('affectedAssets', []))}")
                
            return True
        else:
            logger.error(f"Failed to get anomalies: {response.status_code} {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error testing REST endpoint: {str(e)}")
        return False

async def test_websocket_endpoint():
    """Test the WebSocket endpoint for real-time anomaly updates"""
    try:
        # Connect to the WebSocket
        url = f"{WS_BASE_URL}/orchestrator_ws/anomalies"
        logger.info(f"Connecting to WebSocket endpoint: {url}")
        
        async with websockets.connect(url) as websocket:
            logger.info("Successfully connected to WebSocket")
            
            # Listen for messages for 30 seconds (should be enough to get at least one update)
            timeout = 30
            logger.info(f"Listening for messages for {timeout} seconds...")
            
            start_time = datetime.now()
            message_count = 0
            
            while (datetime.now() - start_time).total_seconds() < timeout:
                try:
                    # Set a timeout for receiving messages
                    message = await asyncio.wait_for(websocket.recv(), timeout=2)
                    message_data = json.loads(message)
                    message_count += 1
                    
                    logger.info(f"Received message #{message_count}:")
                    logger.info(f"  Type: {message_data.get('type', 'unknown')}")
                    
                    if 'data' in message_data:
                        data = message_data['data']
                        logger.info(f"  Anomaly: {data.get('anomalyDetected', 'N/A')}")
                        logger.info(f"  Severity: {data.get('severity', 0)} ({data.get('severityLabel', 'unknown')})")
                        logger.info(f"  Recommended action: {data.get('recommendedAction', 'N/A')}")
                        
                        if 'allAnomalies' in data:
                            logger.info(f"  Total anomalies: {len(data['allAnomalies'])}")
                except asyncio.TimeoutError:
                    # No message received in the timeout period, continue waiting
                    pass
                
            logger.info(f"Test completed. Received {message_count} messages.")
            return True
    except Exception as e:
        logger.error(f"Error testing WebSocket endpoint: {str(e)}")
        return False

async def run_tests():
    """Run all tests"""
    logger.info("Starting anomaly detection endpoint tests")
    
    rest_result = await test_rest_endpoint()
    logger.info(f"REST endpoint test {'passed' if rest_result else 'failed'}")
    
    ws_result = await test_websocket_endpoint()
    logger.info(f"WebSocket endpoint test {'passed' if ws_result else 'failed'}")
    
    logger.info("All tests completed")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_tests())
