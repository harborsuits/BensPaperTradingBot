import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

class NewsAdapterError(Exception):
    pass

class NewsdataAdapter:
    """
    Adapter for Newsdata.io news API.
    """
    BASE_URL = "https://newsdata.io/api/1/news"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWSDATA_API_KEY")
        if not self.api_key:
            raise NewsAdapterError("Newsdata.io API key not provided.")

    def get_news(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        params = {
            "q": symbol,
            "from_date": start_date,
            "to_date": end_date,
            "apikey": self.api_key
        }
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])
        except Exception as e:
            raise NewsAdapterError(f"Newsdata.io news fetch failed: {e}")

    def get_sentiment(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        # Newsdata.io does not provide direct sentiment; placeholder for future NLP
        news = self.get_news(symbol, start_date, end_date)
        return {"symbol": symbol, "sentiment": None, "articles": news}
