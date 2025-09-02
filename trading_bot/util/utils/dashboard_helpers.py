import random
from datetime import datetime

def get_market_data_by_type(data_type="market_indices"):
    """Get different types of market data with error handling"""
    try:
        if data_type == "market_indices":
            # Mock market indices data
            return [
                {"index": "S&P 500", "value": "4,932.25", "change": "+0.8%"},
                {"index": "Dow Jones", "value": "38,750.12", "change": "+0.5%"},
                {"index": "NASDAQ", "value": "16,780.43", "change": "+1.2%"},
                {"index": "Russell 2000", "value": "2,134.81", "change": "+0.3%"}
            ]
        elif data_type == "sector_performance":
            # Mock sector performance data
            return [
                {"sector": "Technology", "change": "+1.2%", "driver": "Semiconductor strength", "holdings": "AAPL, MSFT, NVDA"},
                {"sector": "Energy", "change": "-0.5%", "driver": "Crude oil pressure", "holdings": "XOM, CVX"},
                {"sector": "Healthcare", "change": "+0.7%", "driver": "Pharma earnings", "holdings": "JNJ, PFE"},
                {"sector": "Financials", "change": "+0.4%", "driver": "Rate expectations", "holdings": "JPM, BAC"},
                {"sector": "Consumer Discretionary", "change": "+0.9%", "driver": "Retail sales data", "holdings": "AMZN, HD"}
            ]
        else:
            # Fallback to empty data
            return []
    except Exception as e:
        print(f"Error fetching {data_type} data: {str(e)}")
        return []

def get_news_by_impact(max_items=5, min_impact_score=None, max_impact_score=None):
    """Get news filtered by impact score with error handling"""
    try:
        # Mock news data
        all_news = [
            {
                "source": "Financial Times",
                "time": "10:30 AM",
                "headline": "Fed signals possible rate cut in September meeting",
                "impact_score": 9,
                "summary": "Federal Reserve minutes indicate a shift toward easing monetary policy given cooling inflation data.",
                "action": "Review portfolio interest rate sensitivity."
            },
            {
                "source": "Bloomberg",
                "time": "9:15 AM",
                "headline": "NVIDIA beats earnings expectations by 15%",
                "impact_score": 8,
                "summary": "Chip demand for AI applications continues to drive revenue growth far beyond analyst predictions.",
                "action": "Consider increasing semiconductor exposure."
            },
            {
                "source": "CNBC",
                "time": "11:45 AM",
                "headline": "Consumer confidence index falls to 6-month low",
                "impact_score": 7,
                "summary": "Consumers express concern about future economic conditions despite strong employment data.",
                "action": "Monitor consumer discretionary positions."
            },
            {
                "source": "Reuters",
                "time": "8:30 AM",
                "headline": "Europe manufacturing PMI shows unexpected expansion",
                "impact_score": 6,
                "summary": "European manufacturing sector shows signs of recovery after prolonged contraction.",
                "action": "Consider European exposure for diversification."
            },
            {
                "source": "Wall Street Journal",
                "time": "1:20 PM",
                "headline": "Major tech companies announce new AI partnerships",
                "impact_score": 7,
                "summary": "Strategic partnerships aim to accelerate AI development across multiple industries.",
                "action": "Research AI beneficiaries beyond core tech names."
            },
            {
                "source": "MarketWatch",
                "time": "10:05 AM",
                "headline": "Oil prices drop on larger than expected inventory build",
                "impact_score": 5,
                "summary": "Crude oil inventories show surprise increase, indicating potential demand weakness.",
                "action": "Review energy sector positions."
            }
        ]
        
        # Filter by impact score if provided
        filtered_news = all_news
        if min_impact_score is not None:
            filtered_news = [n for n in filtered_news if n.get('impact_score', 0) >= min_impact_score]
        if max_impact_score is not None:
            filtered_news = [n for n in filtered_news if n.get('impact_score', 0) <= max_impact_score]
            
        # Limit number of items
        return filtered_news[:max_items]
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []
