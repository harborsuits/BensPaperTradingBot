import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

class NewsAdapterError(Exception):
    pass

class GNewsAdapter:
    """
    Adapter for GNews API.
    """
    BASE_URL = "https://gnews.io/api/v4/search"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GNEWS_API_KEY")
        if not self.api_key:
            raise NewsAdapterError("GNews API key not provided.")

    def get_news(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        params = {
            "q": symbol,
            "from": start_date,
            "to": end_date,
            "token": self.api_key
        }
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("articles", [])
        except Exception as e:
            raise NewsAdapterError(f"GNews news fetch failed: {e}")

    def get_sentiment(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        # GNews does not provide direct sentiment; placeholder for future NLP
        news = self.get_news(symbol, start_date, end_date)
        return {"symbol": symbol, "sentiment": None, "articles": news}
