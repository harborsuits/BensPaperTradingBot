#!/usr/bin/env python3
"""
API Integration Checker for BenBot React Frontend

This script checks if all required endpoints for the React frontend integration
are properly implemented in the FastAPI backend.
"""

import os
import sys
import requests
import json
from typing import Dict, List, Any, Set
import logging
from rich.console import Console
from rich.table import Table

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("api_checker")

# Initialize rich console
console = Console()

# API base URL (assuming local development)
API_BASE_URL = "http://localhost:8000"

# Required endpoints for frontend integration
REQUIRED_ENDPOINTS = {
    # Market Context endpoints
    "/api/context": "GET",
    "/api/context/regime": "GET",
    "/api/context/news": "GET",
    "/api/context/news/symbol": "GET",
    "/api/context/features": "GET",
    "/api/context/anomalies": "GET",
    "/api/context/prediction": "GET",
    
    # Strategy Management endpoints
    "/api/strategies": "GET",
    "/api/strategies/{strategy_id}": "GET",
    "/api/strategies/{strategy_id}": "PUT",
    "/api/strategies/ranking": "GET",
    "/api/strategies/insights": "GET",
    
    # Portfolio & Trading endpoints
    "/api/positions": "GET",
    "/api/orders": "GET",
    "/api/trades": "GET",
    "/api/orders": "POST",
    "/api/orders/{order_id}": "DELETE",
    
    # Trade Decisions endpoints
    "/api/decisions/latest": "GET",
    "/api/decisions": "GET",
    
    # Logging & Notifications endpoints
    "/api/logs": "GET",
    "/api/alerts": "GET",
    "/api/system/status": "GET",
    
    # EvoTester endpoints
    "/api/evotester/start": "POST",
    "/api/evotester/{session_id}/stop": "POST",
    "/api/evotester/{session_id}/status": "GET",
    "/api/evotester/{session_id}/result": "GET",
    "/api/evotester/recent": "GET",
    
    # Authentication endpoints
    "/auth/login": "POST",
    
    # WebSocket endpoint
    "/ws": "WS",
}

# Custom User-Agent for the checker
HEADERS = {
    "User-Agent": "BenBot-API-Integration-Checker/1.0",
}

def check_openapi_schema(api_url: str) -> Dict[str, Any]:
    """
    Fetch the OpenAPI schema from the FastAPI backend.
    """
    try:
        response = requests.get(f"{api_url}/openapi.json", headers=HEADERS)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to fetch OpenAPI schema: {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Error fetching OpenAPI schema: {str(e)}")
        return {}

def extract_endpoints_from_schema(schema: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract available endpoints from the OpenAPI schema.
    """
    available_endpoints = {}
    
    if not schema or "paths" not in schema:
        return available_endpoints
    
    for path, methods in schema["paths"].items():
        endpoint_methods = []
        for method in methods:
            if method.lower() != "parameters":  # Skip parameters entry
                endpoint_methods.append(method.upper())
        available_endpoints[path] = endpoint_methods
        
    return available_endpoints

def check_endpoint_availability(required_endpoints: Dict[str, str],
                               available_endpoints: Dict[str, List[str]]) -> Dict[str, bool]:
    """
    Check if required endpoints are available in the API.
    """
    results = {}
    
    for endpoint, method in required_endpoints.items():
        # Handle path parameters
        endpoint_pattern = endpoint
        if "{" in endpoint:
            # Convert {param} to actual regex pattern for matching
            parts = endpoint.split("/")
            for i, part in enumerate(parts):
                if "{" in part and "}" in part:
                    parts[i] = "*"  # Replace with wildcard for matching
            endpoint_pattern = "/".join(parts)
        
        # Find matching endpoint in available_endpoints
        found = False
        matching_endpoint = None
        
        for avail_endpoint, avail_methods in available_endpoints.items():
            # Exact match
            if avail_endpoint == endpoint:
                matching_endpoint = avail_endpoint
                found = method in avail_methods or (method == "WS" and "GET" in avail_methods)
                break
            
            # Pattern match for endpoints with parameters
            elif "{" in endpoint:
                avail_parts = avail_endpoint.split("/")
                req_parts = endpoint_pattern.split("/")
                
                if len(avail_parts) == len(req_parts):
                    match = True
                    for i, (avail_part, req_part) in enumerate(zip(avail_parts, req_parts)):
                        if req_part != "*" and avail_part != req_part:
                            match = False
                            break
                    
                    if match:
                        matching_endpoint = avail_endpoint
                        found = method in avail_methods or (method == "WS" and "GET" in avail_methods)
                        break
        
        results[endpoint] = {
            "found": found,
            "matching_endpoint": matching_endpoint,
            "required_method": method
        }
    
    return results

def display_results(check_results: Dict[str, Any]):
    """
    Display the results in a nice table format.
    """
    # Create a rich table
    table = Table(title="BenBot API Integration Check Results")
    
    # Add columns
    table.add_column("Endpoint", style="cyan")
    table.add_column("Method", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Notes", style="white")
    
    # Count available and missing endpoints
    available_count = 0
    missing_count = 0
    
    # Add rows
    for endpoint, result in check_results.items():
        status = "✅ Available" if result["found"] else "❌ Missing"
        
        if result["found"]:
            available_count += 1
            notes = f"Matched to {result['matching_endpoint']}"
        else:
            missing_count += 1
            notes = "Required for frontend integration"
        
        table.add_row(
            endpoint,
            result["required_method"],
            status,
            notes
        )
    
    # Print the table
    console.print(table)
    
    # Print summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"✅ Available endpoints: {available_count}/{len(check_results)}")
    console.print(f"❌ Missing endpoints: {missing_count}/{len(check_results)}")
    
    # Provide recommendations
    if missing_count > 0:
        console.print("\n[bold yellow]Recommendations:[/bold yellow]")
        console.print("The following endpoints need to be implemented for the React frontend integration:")
        
        for endpoint, result in check_results.items():
            if not result["found"]:
                console.print(f"  - [cyan]{result['required_method']} {endpoint}[/cyan]")
        
        console.print("\nSee the API_DOCUMENTATION.md file for details on these endpoints.")
    else:
        console.print("\n[bold green]All required endpoints are available![/bold green]")

def main():
    """
    Main function to check API integration readiness.
    """
    console.print("[bold]BenBot API Integration Checker[/bold]")
    console.print(f"Checking API at {API_BASE_URL}...\n")
    
    # Check if the API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", headers=HEADERS)
        if response.status_code != 200:
            console.print("[bold red]Error:[/bold red] API server does not appear to be running!")
            console.print(f"Please start the API server by running: [yellow]./start_trading_api.sh[/yellow]")
            return
    except requests.ConnectionError:
        console.print("[bold red]Error:[/bold red] Cannot connect to the API server!")
        console.print(f"Please make sure the API server is running at {API_BASE_URL}")
        return
    
    # Fetch OpenAPI schema
    console.print("Fetching OpenAPI schema...")
    schema = check_openapi_schema(API_BASE_URL)
    
    if not schema:
        console.print("[bold red]Error:[/bold red] Failed to fetch OpenAPI schema!")
        return
    
    # Extract available endpoints
    console.print("Analyzing available endpoints...")
    available_endpoints = extract_endpoints_from_schema(schema)
    
    # Check endpoint availability
    console.print("Checking required endpoints...")
    results = check_endpoint_availability(REQUIRED_ENDPOINTS, available_endpoints)
    
    # Display results
    display_results(results)

if __name__ == "__main__":
    main()
