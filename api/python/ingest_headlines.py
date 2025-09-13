#!/usr/bin/env python3
"""
Headline Ingestion Script for Gemma Embedder

Ingests historical headlines with market outcomes for semantic similarity training.
"""

import json
import csv
import requests
from datetime import datetime, timedelta
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMBEDDER_URL = "http://localhost:8002"

def load_headlines_from_csv(csv_path):
    """Load headlines from CSV file"""
    headlines = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                headlines.append({
                    'id': row.get('id', f"csv_{len(headlines)}"),
                    'text': row.get('headline', row.get('text', '')),
                    'timestamp': row.get('timestamp', row.get('date', datetime.now().isoformat())),
                    'tickers': row.get('tickers', row.get('symbols', '')).split(',') if row.get('tickers') else [],
                    'sector': row.get('sector', 'general'),
                    'sentiment': float(row.get('sentiment', 0.0)) if row.get('sentiment') else None,
                    'outcome_1d': float(row.get('return_1d', 0.0)) if row.get('return_1d') else None,
                    'outcome_5d': float(row.get('return_5d', 0.0)) if row.get('return_5d') else None
                })
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

    return headlines

def fetch_recent_news_from_api():
    """Fetch recent news from the live API"""
    headlines = []
    try:
        # Try to get news from the live API
        response = requests.get("http://localhost:4000/api/news/recent?limit=100", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'headlines' in data:
                for item in data['headlines']:
                    headlines.append({
                        'id': item.get('id', f"api_{len(headlines)}"),
                        'text': item.get('headline', item.get('text', '')),
                        'timestamp': item.get('timestamp', item.get('publishedAt', datetime.now().isoformat())),
                        'tickers': item.get('tickers', item.get('symbols', [])),
                        'sector': item.get('sector', 'general'),
                        'sentiment': item.get('sentiment'),
                        'outcome_1d': None,  # Would need to be calculated
                        'outcome_5d': None   # Would need to be calculated
                    })
    except Exception as e:
        print(f"Could not fetch from API: {e}")

    return headlines

def store_headline_with_embedding(headline):
    """Store a headline with its embedding"""
    try:
        response = requests.post(f"{EMBEDDER_URL}/store", json=headline, timeout=30)
        if response.status_code == 200:
            print(f"✓ Stored: {headline['text'][:50]}...")
            return True
        else:
            print(f"✗ Failed to store: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error storing headline: {e}")
        return False

def main():
    print("🧠 Gemma Headline Ingestion")
    print("===========================")

    # Check if embedder is running
    try:
        response = requests.get(f"{EMBEDDER_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Embedder service not running. Start with: ./start_embedder.sh")
            return
    except:
        print("❌ Cannot connect to embedder service. Start with: ./start_embedder.sh")
        return

    print("✅ Embedder service is running")

    # Get headlines from various sources
    headlines = []

    # Try to load from CSV if provided
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if os.path.exists(csv_path):
            print(f"📄 Loading headlines from CSV: {csv_path}")
            headlines.extend(load_headlines_from_csv(csv_path))
        else:
            print(f"❌ CSV file not found: {csv_path}")

    # Try to fetch from API
    print("🌐 Fetching recent news from API...")
    api_headlines = fetch_recent_news_from_api()
    headlines.extend(api_headlines)

    if not headlines:
        print("❌ No headlines found. Provide a CSV file or ensure API has news data.")
        print("💡 CSV format: id,headline,timestamp,tickers,sector,sentiment,return_1d,return_5d")
        return

    print(f"📊 Found {len(headlines)} headlines to process")

    # Process headlines in batches
    batch_size = 10
    stored_count = 0

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i+batch_size]
        print(f"📦 Processing batch {i//batch_size + 1}/{(len(headlines) + batch_size - 1)//batch_size}")

        for headline in batch:
            if store_headline_with_embedding(headline):
                stored_count += 1

            # Small delay to avoid overwhelming the service
            time.sleep(0.1)

    print("
🎉 Ingestion complete!"    print(f"✅ Stored {stored_count}/{len(headlines)} headlines")
    print(f"🔍 You can now query similar cases via POST {EMBEDDER_URL}/similar")

if __name__ == "__main__":
    main()
