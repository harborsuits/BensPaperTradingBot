from fastapi import APIRouter, Query
from trading_bot.services.news_service import sentiment_by_category

router = APIRouter(prefix="/api/news", tags=["news"])

@router.get("/sentiment")
async def news_sentiment(category: str = Query("markets"), query: str = "", per_source: int = 5):
    """
    Fuses multiple outlets for a category, clusters near-duplicates, and returns
    per-cluster sentiment + partisanship + info density + finance relevance.
    
    Parameters:
    - category: News category (markets, politics, tech, crypto, macro)
    - query: Optional filter for headlines containing this text
    - per_source: Maximum number of articles to process per source
    
    Returns:
    - Category name
    - Clusters of similar articles with aggregated metrics
    - Per-outlet statistics
    """
    data = await sentiment_by_category(category=category, query=query, per_source=per_source)
    return data