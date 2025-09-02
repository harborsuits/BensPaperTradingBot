from typing import Dict, Type, Any
from trading_bot.adapters.news.finnhub_adapter import FinnhubNewsAdapter
from trading_bot.adapters.news.nytimes_adapter import NYTimesNewsAdapter
from trading_bot.adapters.news.newsdata_adapter import NewsdataAdapter
from trading_bot.adapters.news.gnews_adapter import GNewsAdapter
from trading_bot.adapters.news.mediastack_adapter import MediastackAdapter
from trading_bot.adapters.news.currents_adapter import CurrentsAdapter
from trading_bot.adapters.ai.llm_adapter import LLMAdapter

class AdapterRegistry:
    """
    Central registry for all provider adapters.
    """
    NEWS_PROVIDERS = {
        "finnhub": FinnhubNewsAdapter,
        "nytimes": NYTimesNewsAdapter,
        "newsdata": NewsdataAdapter,
        "gnews": GNewsAdapter,
        "mediastack": MediastackAdapter,
        "currents": CurrentsAdapter
    }
    AI_PROVIDERS = {
        "openai": lambda: LLMAdapter(provider="openai"),
        "claude": lambda: LLMAdapter(provider="claude")
    }
    # Extend for market/trading providers as needed

    @staticmethod
    def get_news_adapter(provider: str, **kwargs) -> Any:
        if provider not in AdapterRegistry.NEWS_PROVIDERS:
            raise ValueError(f"Unknown news provider: {provider}")
        return AdapterRegistry.NEWS_PROVIDERS[provider](**kwargs)

    @staticmethod
    def get_ai_adapter(provider: str, **kwargs) -> Any:
        if provider not in AdapterRegistry.AI_PROVIDERS:
            raise ValueError(f"Unknown AI provider: {provider}")
        return AdapterRegistry.AI_PROVIDERS[provider]()
