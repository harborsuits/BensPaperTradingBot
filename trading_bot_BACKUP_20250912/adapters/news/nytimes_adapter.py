import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

class NewsAdapterError(Exception):
    pass

class NYTimesNewsAdapter:
    """
    Adapter for New York Times news API.
    """
    BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NYTIMES_API_KEY")
        if not self.api_key:
            raise NewsAdapterError("NYTimes API key not provided.")

    def get_news(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        # NYTimes uses 'q' for query, 'begin_date'/'end_date' in YYYYMMDD
        params = {
            "q": symbol,
            "begin_date": start_date.replace('-', ''),
            "end_date": end_date.replace('-', ''),
            "api-key": self.api_key
        }
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", {}).get("docs", [])
        except Exception as e:
            raise NewsAdapterError(f"NYTimes news fetch failed: {e}")

    def get_sentiment(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        # NYTimes does not provide direct sentiment; placeholder for future NLP
        news = self.get_news(symbol, start_date, end_date)
        # Could run NLP sentiment analysis on headlines/snippets here
        return {"symbol": symbol, "sentiment": None, "articles": news}
