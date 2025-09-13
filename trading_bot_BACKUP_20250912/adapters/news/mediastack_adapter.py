import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

class NewsAdapterError(Exception):
    pass

class MediastackAdapter:
    """
    Adapter for Mediastack API.
    """
    BASE_URL = "http://api.mediastack.com/v1/news"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MEDIASTACK_API_KEY")
        if not self.api_key:
            raise NewsAdapterError("Mediastack API key not provided.")

    def get_news(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        params = {
            "access_key": self.api_key,
            "keywords": symbol,
            "date": f"{start_date},{end_date}"
        }
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])
        except Exception as e:
            raise NewsAdapterError(f"Mediastack news fetch failed: {e}")

    def get_sentiment(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        # Mediastack does not provide direct sentiment; placeholder for future NLP
        news = self.get_news(symbol, start_date, end_date)
        return {"symbol": symbol, "sentiment": None, "articles": news}
