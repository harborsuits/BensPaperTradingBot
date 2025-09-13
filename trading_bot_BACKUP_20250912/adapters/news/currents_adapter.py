import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

class NewsAdapterError(Exception):
    pass

class CurrentsAdapter:
    """
    Adapter for Currents API.
    """
    BASE_URL = "https://api.currentsapi.services/v1/search"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("CURRENTS_API_KEY")
        if not self.api_key:
            raise NewsAdapterError("Currents API key not provided.")

    def get_news(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        params = {
            "keywords": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "apiKey": self.api_key
        }
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("news", [])
        except Exception as e:
            raise NewsAdapterError(f"Currents API news fetch failed: {e}")

    def get_sentiment(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        # Currents API does not provide direct sentiment; placeholder for future NLP
        news = self.get_news(symbol, start_date, end_date)
        return {"symbol": symbol, "sentiment": None, "articles": news}
