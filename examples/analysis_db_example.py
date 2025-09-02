#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script demonstrating how to use the analysis database.

This script showcases the basic operations of the AnalysisDatabase class
including saving and retrieving various types of analysis results and selection history.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_bot.data.analysis_db import AnalysisDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("analysis_db_example")

def main():
    """Run the example script."""
    
    # Create a new database instance with a custom path for this example
    db_path = os.path.join(os.path.dirname(__file__), "example_analysis.db")
    db = AnalysisDatabase(db_path)
    logger.info(f"Created example database at {db_path}")
    
    # Example symbols to use
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    # 1. Save stock analysis results
    logger.info("Saving stock analysis results...")
    for symbol in symbols:
        # Save technical analysis
        technical_score = random.uniform(0, 1)
        technical_metrics = {
            "rsi": random.uniform(30, 70),
            "macd": random.uniform(-2, 2),
            "bollinger_bands": random.uniform(-1, 1),
            "volume_trend": random.choice(["increasing", "decreasing", "stable"]),
        }
        
        db.save_stock_analysis(
            symbol=symbol,
            analysis_type="technical",
            score=technical_score,
            rank=symbols.index(symbol) + 1,
            recommendation="buy" if technical_score > 0.7 else "hold" if technical_score > 0.4 else "sell",
            metrics=technical_metrics,
            analysis_details={
                "signal_strength": "strong" if technical_score > 0.8 else "moderate",
                "trend": "bullish" if technical_score > 0.6 else "bearish",
                "support_level": random.uniform(80, 120),
                "resistance_level": random.uniform(120, 160)
            },
            model_version="1.0.0"
        )
        
        # Save fundamental analysis
        fundamental_score = random.uniform(0, 1)
        fundamental_metrics = {
            "pe_ratio": random.uniform(10, 30),
            "eps_growth": random.uniform(-0.1, 0.3),
            "revenue_growth": random.uniform(-0.05, 0.2),
            "debt_to_equity": random.uniform(0.2, 1.5),
            "profit_margin": random.uniform(0.05, 0.25)
        }
        
        db.save_stock_analysis(
            symbol=symbol,
            analysis_type="fundamental",
            score=fundamental_score,
            recommendation="buy" if fundamental_score > 0.7 else "hold" if fundamental_score > 0.4 else "sell",
            metrics=fundamental_metrics,
            analysis_details={
                "valuation": "undervalued" if fundamental_score > 0.7 else "fair" if fundamental_score > 0.4 else "overvalued",
                "growth_prospects": "high" if fundamental_score > 0.6 else "moderate" if fundamental_score > 0.3 else "low",
                "financial_health": "strong" if fundamental_score > 0.5 else "concerning",
            },
            model_version="2.1.0"
        )
        
        # Save sentiment analysis for each symbol
        for source in ["twitter", "news", "reddit"]:
            sentiment_score = random.uniform(-1, 1)
            sentiment_label = "positive" if sentiment_score > 0.3 else "neutral" if sentiment_score > -0.3 else "negative"
            
            db.save_sentiment_analysis(
                symbol=symbol,
                source=source,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                volume=random.randint(100, 5000),
                key_phrases=["earnings", "product launch", "market share", "competition"] if random.random() > 0.5 else [],
                source_details={
                    "influencer_count": random.randint(0, 10),
                    "trending": random.random() > 0.8,
                    "controversy_level": random.choice(["low", "medium", "high"])
                }
            )
    
    # 2. Save stock selection history
    logger.info("Saving stock selection history...")
    
    # Selection based on technical analysis
    selected_symbols = sorted(symbols, key=lambda s: random.random())[:3]
    weights = {symbol: 1.0/len(selected_symbols) for symbol in selected_symbols}
    
    db.save_stock_selection(
        selection_criteria="technical_momentum",
        symbols=selected_symbols,
        weights=weights,
        market_regime="bullish",
        reasoning="Selected top momentum stocks based on technical indicators",
        performance_snapshot={
            "expected_return": random.uniform(0.05, 0.2),
            "expected_volatility": random.uniform(0.1, 0.3),
            "sharpe_ratio": random.uniform(0.8, 2.5),
            "max_drawdown": random.uniform(0.05, 0.2)
        }
    )
    
    # Selection based on fundamental analysis
    selected_symbols = sorted(symbols, key=lambda s: random.random())[:2]
    weights = {symbol: 1.0/len(selected_symbols) for symbol in selected_symbols}
    
    db.save_stock_selection(
        selection_criteria="value_investing",
        symbols=selected_symbols,
        weights=weights,
        market_regime="neutral",
        reasoning="Selected undervalued stocks with strong fundamentals",
        performance_snapshot={
            "expected_return": random.uniform(0.03, 0.15),
            "expected_volatility": random.uniform(0.08, 0.25),
            "sharpe_ratio": random.uniform(0.6, 2.0),
            "max_drawdown": random.uniform(0.03, 0.15)
        }
    )
    
    # 3. Save strategy selection history
    logger.info("Saving strategy selection history...")
    
    strategies = [
        "momentum_strategy", 
        "value_strategy", 
        "mean_reversion", 
        "trend_following", 
        "options_income"
    ]
    
    for strategy in strategies:
        confidence = random.uniform(0.4, 0.95)
        
        db.save_strategy_selection(
            selected_strategy=strategy,
            market_regime=random.choice(["bullish", "bearish", "neutral", "volatile"]),
            confidence_score=confidence,
            reasoning=f"Selected based on current market conditions with {confidence:.2f} confidence",
            parameters={
                "lookback_period": random.randint(10, 60),
                "entry_threshold": random.uniform(0.1, 0.5),
                "exit_threshold": random.uniform(0.1, 0.5),
                "stop_loss": random.uniform(0.05, 0.2),
                "position_size": random.uniform(0.05, 0.25)
            }
        )
    
    # 4. Save market regime analysis
    logger.info("Saving market regime analysis...")
    
    regimes = ["bullish", "bearish", "neutral", "volatile", "trending", "range-bound"]
    for _ in range(3):
        regime = random.choice(regimes)
        confidence = random.uniform(0.5, 0.95)
        
        db.save_market_regime(
            regime=regime,
            confidence=confidence,
            indicators={
                "vix": random.uniform(10, 40),
                "atr": random.uniform(1, 10),
                "market_breadth": random.uniform(-1, 1),
                "trend_strength": random.uniform(0, 1),
                "momentum": random.uniform(-1, 1),
                "correlation": random.uniform(-1, 1)
            },
            description=f"Market is currently in a {regime} regime with {confidence:.2f} confidence"
        )
    
    # 5. Query and display the data
    logger.info("\nQuerying the database...")
    
    # Get the latest technical analysis for AAPL
    logger.info("Latest technical analysis for AAPL:")
    tech_analysis = db.get_latest_stock_analysis("AAPL", "technical")
    if tech_analysis:
        for key, value in tech_analysis[0].items():
            if key not in ["metrics", "analysis_details"]:
                logger.info(f"  {key}: {value}")
        
        logger.info("  Technical metrics:")
        for metric, value in tech_analysis[0]["metrics"].items():
            logger.info(f"    {metric}: {value}")
    
    # Get stock selection history
    logger.info("\nStock selection history:")
    selections = db.get_stock_selection_history(limit=5)
    for selection in selections:
        logger.info(f"  {selection['timestamp']} - {selection['selection_criteria']} - Symbols: {', '.join(selection['symbols'])}")
    
    # Get strategy selection history
    logger.info("\nStrategy selection history:")
    strategy_history = db.get_strategy_selection_history(limit=5)
    for entry in strategy_history:
        logger.info(f"  {entry['timestamp']} - {entry['selected_strategy']} - {entry['market_regime']} - Confidence: {entry['confidence_score']}")
    
    # Get market regime history
    logger.info("\nMarket regime history:")
    regime_history = db.get_market_regime_history(limit=3)
    for entry in regime_history:
        logger.info(f"  {entry['timestamp']} - {entry['regime']} - Confidence: {entry['confidence']}")
        logger.info(f"    Key indicators: {', '.join(entry['indicators'].keys())}")
    
    # Get sentiment analysis for MSFT
    logger.info("\nSentiment analysis for MSFT:")
    sentiment = db.get_sentiment_analysis("MSFT")
    for entry in sentiment:
        logger.info(f"  {entry['timestamp']} - {entry['source']} - Score: {entry['sentiment_score']} ({entry['sentiment_label']})")
        if entry['key_phrases']:
            logger.info(f"    Key phrases: {', '.join(entry['key_phrases'])}")
    
    logger.info(f"\nExample completed. Database saved at {db_path}")
    logger.info("Run this example again to add more data to the database.")

if __name__ == "__main__":
    main() 