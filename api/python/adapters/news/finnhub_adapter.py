import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

class NewsAdapterError(Exception):
    pass

class FinnhubNewsAdapter:
    """
    Adapter for Finnhub news and sentiment API.
    """
    BASE_URL = "https://finnhub.io/api/v1/"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise NewsAdapterError("Finnhub API key not provided.")

    def get_news(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        url = f"{self.BASE_URL}company-news"
        params = {
            "symbol": symbol,
            "from": start_date,
            "to": end_date,
            "token": self.api_key
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise NewsAdapterError(f"Finnhub news fetch failed: {e}")

    def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}stock/social-sentiment"
        params = {"symbol": symbol, "token": self.api_key}
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise NewsAdapterError(f"Finnhub sentiment fetch failed: {e}")
