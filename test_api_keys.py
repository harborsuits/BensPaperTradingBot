#!/usr/bin/env python3
"""
Simple API key verification script for BenBot trading system
"""

import os
import requests
import sys

def load_env_file():
    """Load environment variables from .env file"""
    env_file = '/Users/bendickinson/Desktop/benbot/.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

def test_finnhub_api():
    """Test Finnhub API connection"""
    api_key = os.getenv('FINNHUB_KEY')
    if not api_key:
        print("❌ FINNHUB_KEY not found")
        return False
    
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'c' in data:  # Current price exists
                print(f"✅ Finnhub API: Connected - AAPL current price: ${data['c']}")
                return True
        print(f"❌ Finnhub API: Failed with status {response.status_code}")
        return False
    except Exception as e:
        print(f"❌ Finnhub API: Error - {str(e)}")
        return False

def test_alpha_vantage_api():
    """Test Alpha Vantage API connection"""
    api_key = os.getenv('ALPHA_VANTAGE_KEY')
    if not api_key:
        print("❌ ALPHA_VANTAGE_KEY not found")
        return False
    
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'Global Quote' in data:
                price = data['Global Quote'].get('05. price', 'N/A')
                print(f"✅ Alpha Vantage API: Connected - AAPL price: ${price}")
                return True
        print(f"❌ Alpha Vantage API: Failed with status {response.status_code}")
        return False
    except Exception as e:
        print(f"❌ Alpha Vantage API: Error - {str(e)}")
        return False

def test_news_api():
    """Test News API connection (MediaStack)"""
    api_key = os.getenv('MEDIASTACK_KEY')
    if not api_key:
        print("❌ MEDIASTACK_KEY not found")
        return False
    
    try:
        url = f"http://api.mediastack.com/v1/news?access_key={api_key}&keywords=AAPL&limit=1"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                print(f"✅ MediaStack News API: Connected - Found {len(data['data'])} articles")
                return True
        print(f"❌ MediaStack News API: Failed with status {response.status_code}")
        return False
    except Exception as e:
        print(f"❌ MediaStack News API: Error - {str(e)}")
        return False

def main():
    # Load environment variables from .env file
    load_env_file()
    
    print("🔍 Testing API Keys for BenBot Trading System")
    print("=" * 50)
    
    results = []
    
    # Test Market Data APIs
    print("\n📊 Testing Market Data APIs:")
    results.append(("Finnhub", test_finnhub_api()))
    results.append(("Alpha Vantage", test_alpha_vantage_api()))
    
    # Test News APIs
    print("\n📰 Testing News APIs:")
    results.append(("MediaStack", test_news_api()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY:")
    working = sum(1 for _, result in results if result)
    total = len(results)
    print(f"✅ Working: {working}/{total}")
    
    if working == total:
        print("🎉 All tested APIs are working!")
        return 0
    else:
        print("⚠️  Some APIs failed. Check your keys and network connection.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
