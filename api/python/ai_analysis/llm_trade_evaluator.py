"""
LLM-Enhanced Trade Evaluator

This module uses Large Language Models (LLMs) to evaluate trade setups,
incorporating market sentiment, news, and technical analysis to provide
confidence scores and insights for potential trades.
"""

import os
import json
import logging
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np
import requests
from openai import OpenAI

# Configure logging
logger = logging.getLogger(__name__)

class LLMTradeEvaluator:
    """
    LLM-powered trade evaluator for analyzing and scoring trade setups
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        config_path: Optional[str] = None,
        use_mock: bool = False,
        cache_dir: Optional[str] = None,
        integrator: Optional[Any] = None
    ):
        """
        Initialize the LLM Trade Evaluator.
        
        Args:
            api_key: OpenAI API key
            model: LLM model to use
            config_path: Path to configuration file
            use_mock: Whether to use mock responses instead of API calls
            cache_dir: Directory to cache responses
            integrator: Optional reference to IndicatorSentimentIntegrator for enhanced analysis
        """
        # Load configuration
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Set API key from config, parameter, or environment
        self.api_key = api_key or self.config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        
        if not self.api_key and not use_mock:
            raise ValueError("OpenAI API key not provided")
        
        # Set model name
        self.model = model or self.config.get('model', 'gpt-4')
        
        # Initialize client if not using mock
        self.client = None
        if not use_mock:
            self.client = OpenAI(api_key=self.api_key)
        
        # Set up caching
        self.use_mock = use_mock
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize evaluation tracking
        self.evaluations = []
        
        # Store reference to indicator-sentiment integrator if provided
        self.integrator = integrator
        
        logger.info(f"LLM Trade Evaluator initialized (model: {model}, mock: {use_mock}, integrator: {integrator is not None})")
    
    def evaluate_trade(
        self,
        symbol: str,
        direction: str,
        strategy: str,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        market_data: Optional[Dict[str, Any]] = None,
        news_data: Optional[List[Dict[str, Any]]] = None,
        technical_indicators: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None,
        use_integrated_data: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a potential trade setup using LLM.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction ('long' or 'short')
            strategy: Strategy name
            price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit target price
            market_data: Recent OHLCV data
            news_data: Recent news articles
            technical_indicators: Technical indicators
            market_context: Broader market context
            use_integrated_data: Whether to use the integrated indicator-sentiment data if available
            
        Returns:
            Evaluation results including confidence score and analysis
        """
        cache_key = f"eval_{symbol}_{direction}_{strategy}_{price}_{int(time.time() // 3600)}"
        
        # Try to load from cache
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        logger.debug(f"Loading evaluation from cache: {cache_file}")
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Error loading cache: {str(e)}")
        
        # If integrator is available and integrated data is preferred, use that first
        integrated_data = None
        if self.integrator and use_integrated_data:
            try:
                integrated_data = self.integrator.get_integrated_analysis(symbol)
                if integrated_data:
                    logger.info(f"Using integrated indicator-sentiment data for {symbol}")
                    
                    # Update technical indicators with normalized values if not provided
                    if (not technical_indicators or len(technical_indicators) < 3) and 'normalized_indicators' in integrated_data:
                        technical_indicators = integrated_data.get('normalized_indicators', {})
                    
                    # Add sentiment data if we don't have news data
                    if not news_data and 'normalized_sentiment' in integrated_data:
                        sentiment_data = integrated_data.get('normalized_sentiment', {})
                        # Convert sentiment data to a format similar to news_data
                        if 'news_sentiment' in sentiment_data:
                            mock_news = [{"title": f"Aggregated News Sentiment for {symbol}", 
                                         "sentiment": sentiment_data.get('news_sentiment', 0),
                                         "confidence": sentiment_data.get('confidence', 0.5),
                                         "source": "Integrated Analysis"}]
                            news_data = mock_news if not news_data else news_data + mock_news
            except Exception as e:
                logger.warning(f"Error getting integrated data for {symbol}: {str(e)}")
        
        # If using mock, return mock evaluation
        if self.use_mock:
            evaluation = self._generate_mock_evaluation(
                symbol, direction, strategy, price, stop_loss, take_profit,
                market_data, technical_indicators, market_context
            )
        else:
            # Generate market and news summaries
            market_summary = self._prepare_market_summary(symbol, market_data, technical_indicators, market_context)
            news_summary = self._prepare_news_summary(symbol, news_data)
            
            # Add integrated data summary if available
            integrated_summary = ""
            if integrated_data:
                integrated_summary = self._prepare_integrated_data_summary(integrated_data)
                market_summary += "\n" + integrated_summary
            
            # Generate evaluation
            evaluation = self._generate_llm_evaluation(
                symbol, direction, strategy, price, stop_loss, take_profit,
                market_summary, news_summary
            )
        
        # Add metadata
        evaluation.update({
            "symbol": symbol,
            "direction": direction,
            "strategy": strategy,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "timestamp": datetime.now().isoformat()
        })
        
        # Add integrated data metrics if available
        if integrated_data:
            evaluation["integrated_data"] = {
                "integrated_score": integrated_data.get("integrated_score", 0),
                "confidence": integrated_data.get("confidence", 0.5),
                "indicator_contribution": integrated_data.get("indicator_contribution", 0),
                "sentiment_contribution": integrated_data.get("sentiment_contribution", 0),
                "bias": integrated_data.get("bias", "neutral")
            }
        
        # Add to evaluations history
        self.evaluations.append(evaluation)
        if len(self.evaluations) > 100:  # Keep only the most recent evaluations
            self.evaluations = self.evaluations[-100:]
        
        # Cache if enabled
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            try:
                with open(cache_file, 'w') as f:
                    json.dump(evaluation, f)
                    logger.debug(f"Cached evaluation to: {cache_file}")
            except Exception as e:
                logger.warning(f"Error caching evaluation: {str(e)}")
        
        return evaluation
    
    def _prepare_market_summary(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]] = None,
        technical_indicators: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Prepare a summary of market data for the LLM prompt.
        
        Args:
            symbol: Trading symbol
            market_data: Recent OHLCV data
            technical_indicators: Technical indicators
            market_context: Broader market context
            
        Returns:
            Text summary of market data
        """
        summary = [f"### Market Summary for {symbol}"]
        
        # Add price data if available
        if market_data:
            price_data = market_data.get(symbol, {})
            if price_data:
                summary.append("\n#### Price Data:")
                
                current_price = price_data.get('close', price_data.get('price', 0))
                summary.append(f"Current Price: ${current_price:.2f}")
                
                # Add OHLC if available
                if all(k in price_data for k in ['open', 'high', 'low']):
                    summary.append(f"Open: ${price_data['open']:.2f}, High: ${price_data['high']:.2f}, Low: ${price_data['low']:.2f}")
                
                # Add volume if available
                if 'volume' in price_data:
                    summary.append(f"Volume: {price_data['volume']:,}")
        
        # Add technical indicators if available
        if technical_indicators:
            indicators = technical_indicators.get(symbol, {})
            if indicators:
                summary.append("\n#### Technical Indicators:")
                
                # Format moving averages
                ma_keys = [k for k in indicators if 'ma' in k.lower() or 'moving' in k.lower()]
                if ma_keys:
                    ma_text = []
                    for k in ma_keys:
                        ma_text.append(f"{k}: {indicators[k]:.2f}")
                    summary.append(f"Moving Averages: {', '.join(ma_text)}")
                
                # Format oscillators
                osc_keys = [k for k in indicators if any(o in k.lower() for o in ['rsi', 'macd', 'cci', 'stoch'])]
                if osc_keys:
                    osc_text = []
                    for k in osc_keys:
                        osc_text.append(f"{k}: {indicators[k]:.2f}")
                    summary.append(f"Oscillators: {', '.join(osc_text)}")
                
                # Format trend indicators
                trend_keys = [k for k in indicators if any(t in k.lower() for o in ['trend', 'atr', 'adx'])]
                if trend_keys:
                    trend_text = []
                    for k in trend_keys:
                        trend_text.append(f"{k}: {indicators[k]:.2f}")
                    summary.append(f"Trend Indicators: {', '.join(trend_text)}")
                
                # Add remaining indicators
                other_keys = [k for k in indicators if k not in ma_keys + osc_keys + trend_keys]
                if other_keys:
                    other_text = []
                    for k in other_keys:
                        other_text.append(f"{k}: {indicators[k]}")
                    summary.append(f"Other Indicators: {', '.join(other_text)}")
        
        # Add market context if available
        if market_context:
            summary.append("\n#### Market Context:")
            
            # Add market regime if available
            if 'market_regime' in market_context:
                summary.append(f"Market Regime: {market_context['market_regime']}")
            
            # Add volatility if available
            if 'volatility_index' in market_context:
                summary.append(f"Volatility Index: {market_context['volatility_index']:.2f}")
            
            # Add sector performance if available
            if 'sector_performance' in market_context:
                sector_perf = market_context['sector_performance']
                if isinstance(sector_perf, dict) and sector_perf:
                    # Find best and worst sectors
                    sorted_sectors = sorted(sector_perf.items(), key=lambda x: x[1], reverse=True)
                    best_sector = sorted_sectors[0]
                    worst_sector = sorted_sectors[-1]
                    
                    summary.append(f"Best Performing Sector: {best_sector[0]} ({best_sector[1]:.2f}%)")
                    summary.append(f"Worst Performing Sector: {worst_sector[0]} ({worst_sector[1]:.2f}%)")
        
        return "\n".join(summary)
    
    def _prepare_news_summary(
        self,
        symbol: str,
        news_data: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Prepare a summary of recent news for the LLM prompt.
        
        Args:
            symbol: Trading symbol
            news_data: Recent news articles
            
        Returns:
            Text summary of news data
        """
        summary = [f"### Recent News for {symbol}"]
        
        if not news_data:
            summary.append("No recent news available.")
            return "\n".join(summary)
        
        # Format news articles
        for i, article in enumerate(news_data[:5]):  # Limit to 5 articles
            title = article.get('title', 'Untitled')
            date = article.get('date', 'Unknown date')
            source = article.get('source', 'Unknown source')
            summary.append(f"\n#### {i+1}. {title} ({source}, {date})")
            
            # Add short content if available
            if 'summary' in article:
                summary.append(article['summary'])
            elif 'content' in article:
                # Truncate content to 200 characters
                content = article['content']
                if len(content) > 200:
                    content = content[:200] + "..."
                summary.append(content)
        
        return "\n".join(summary)
    
    def _prepare_integrated_data_summary(self, integrated_data: Dict[str, Any]) -> str:
        """
        Prepare a summary of integrated indicator and sentiment data for the LLM prompt.
        
        Args:
            integrated_data: Integrated indicator and sentiment data from the integrator
            
        Returns:
            A text summary of the integrated data
        """
        summary = "\n=== INTEGRATED INDICATOR AND SENTIMENT ANALYSIS ===\n"
        
        # Extract key metrics
        integrated_score = integrated_data.get("integrated_score", 0)
        indicator_contribution = integrated_data.get("indicator_contribution", 0)
        sentiment_contribution = integrated_data.get("sentiment_contribution", 0)
        confidence = integrated_data.get("confidence", 0.5)
        bias = integrated_data.get("bias", "neutral")
        
        # Interpret the integrated score
        if integrated_score > 0.5:
            bias_text = "strongly bullish"
        elif integrated_score > 0.2:
            bias_text = "moderately bullish"
        elif integrated_score < -0.5:
            bias_text = "strongly bearish"
        elif integrated_score < -0.2:
            bias_text = "moderately bearish"
        else:
            bias_text = "neutral"
        
        summary += f"Overall Bias: {bias_text} (score: {integrated_score:.2f})\n"
        summary += f"Confidence: {confidence:.2f} (scale 0-1)\n"
        summary += f"Technical Indicator Contribution: {indicator_contribution:.2f}\n"
        summary += f"Sentiment Contribution: {sentiment_contribution:.2f}\n\n"
        
        # Add details about the individual components if available
        if "indicator_details" in integrated_data:
            summary += "Technical Indicators:\n"
            for indicator, value in integrated_data["indicator_details"].items():
                if isinstance(value, (int, float)):
                    summary += f"- {indicator}: {value:.2f}\n"
                else:
                    summary += f"- {indicator}: {value}\n"
        
        if "sentiment_details" in integrated_data:
            summary += "\nSentiment Analysis:\n"
            for source, value in integrated_data["sentiment_details"].items():
                if isinstance(value, (int, float)):
                    summary += f"- {source}: {value:.2f}\n"
                else:
                    summary += f"- {source}: {value}\n"
                
        # Add data sources if available
        if "data_sources" in integrated_data:
            sources = integrated_data["data_sources"]
            indicators = sources.get("indicators", [])
            sentiment = sources.get("sentiment", [])
            if indicators or sentiment:
                summary += "\nData Sources Used:\n"
                if indicators:
                    summary += f"- Indicators: {', '.join(indicators)}\n"
                if sentiment:
                    summary += f"- Sentiment: {', '.join(sentiment)}\n"
        
        return summary
    
    def _generate_llm_evaluation(
        self,
        symbol: str,
        direction: str,
        strategy: str,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        market_summary: str = "",
        news_summary: str = ""
    ) -> Dict[str, Any]:
        """
        Generate a trade evaluation using the LLM.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction ('long' or 'short')
            strategy: Strategy name
            price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit target price
            market_summary: Summary of market data
            news_summary: Summary of news data
            
        Returns:
            Evaluation results including confidence score and analysis
        """
        try:
            # Calculate risk/reward if stop loss and take profit provided
            risk_reward = "N/A"
            if stop_loss and take_profit and price:
                if direction == 'long':
                    reward = take_profit - price
                    risk = price - stop_loss
                else:  # short
                    reward = price - take_profit
                    risk = stop_loss - price
                
                if risk > 0:
                    risk_reward = f"{reward / risk:.2f}"
            
            # Construct prompt
            prompt = f"""
You are an expert trading analyst evaluating a potential trade. Your task is to assess the trade setup and provide:
1. A confidence score (0-100) for the trade
2. Analysis of the technical setup
3. Analysis of the news sentiment
4. Overall trade recommendation
5. Key risks to monitor

Trade details:
- Symbol: {symbol}
- Direction: {direction}
- Strategy: {strategy}
- Entry Price: ${price:.2f}
- Stop Loss: ${stop_loss:.2f if stop_loss else 'Not specified'}
- Take Profit: ${take_profit:.2f if take_profit else 'Not specified'}
- Risk/Reward Ratio: {risk_reward}

{market_summary}

{news_summary}

Please structure your response as a JSON object with the following keys:
- confidence_score: a number between 0-100
- technical_analysis: a string with your assessment of the technical setup
- news_sentiment: a string with your assessment of news sentiment
- trade_recommendation: a string with your overall recommendation
- key_risks: an array of strings with key risks to monitor
- suggested_adjustments: an array of strings with suggested adjustments (if any)
            """
            
            # Call the LLM API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert trading analyst providing structured trade evaluations. You always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={ "type": "json_object" }
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            evaluation = json.loads(response_text)
            
            # Ensure required fields are present
            required_fields = ['confidence_score', 'technical_analysis', 'news_sentiment', 
                              'trade_recommendation', 'key_risks']
            for field in required_fields:
                if field not in evaluation:
                    evaluation[field] = "Not provided"
            
            # Convert confidence score to number if needed
            if isinstance(evaluation['confidence_score'], str):
                # Extract numeric portion
                match = re.search(r'\d+', evaluation['confidence_score'])
                if match:
                    evaluation['confidence_score'] = int(match.group())
                else:
                    evaluation['confidence_score'] = 50  # Default if parsing fails
            
            # Ensure key_risks is a list
            if not isinstance(evaluation['key_risks'], list):
                evaluation['key_risks'] = [evaluation['key_risks']]
            
            # Add suggested adjustments if not present
            if 'suggested_adjustments' not in evaluation:
                evaluation['suggested_adjustments'] = []
            
            # Ensure suggested_adjustments is a list
            if not isinstance(evaluation['suggested_adjustments'], list):
                evaluation['suggested_adjustments'] = [evaluation['suggested_adjustments']]
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error generating LLM evaluation: {str(e)}")
            
            # Return default evaluation on error
            return {
                'confidence_score': 50,
                'technical_analysis': f"Error generating evaluation: {str(e)}",
                'news_sentiment': "Not available due to error",
                'trade_recommendation': "Unable to provide recommendation due to error",
                'key_risks': ["Error in evaluation process"],
                'suggested_adjustments': []
            }
    
    def _generate_mock_evaluation(
        self,
        symbol: str,
        direction: str,
        strategy: str,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        market_data: Optional[Dict[str, Any]] = None,
        technical_indicators: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a mock trade evaluation (for testing).
        
        Args:
            symbol: Trading symbol
            direction: Trade direction ('long' or 'short')
            strategy: Strategy name
            price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit target price
            market_data: Recent OHLCV data
            technical_indicators: Technical indicators
            market_context: Broader market context
            
        Returns:
            Mock evaluation results
        """
        # Generate a pseudo-random confidence score based on inputs
        seed = sum(ord(c) for c in symbol) + hash(direction) % 100 + hash(strategy) % 100
        np.random.seed(seed)
        
        # Base confidence score
        confidence = np.random.randint(40, 80)
        
        # Adjust based on market regime if available
        if market_context and 'market_regime' in market_context:
            regime = market_context['market_regime']
            if direction == 'long' and regime in ['bullish', 'moderately_bullish']:
                confidence += np.random.randint(5, 15)
            elif direction == 'short' and regime in ['bearish', 'moderately_bearish']:
                confidence += np.random.randint(5, 15)
            elif direction == 'long' and regime in ['bearish', 'volatile']:
                confidence -= np.random.randint(5, 15)
            elif direction == 'short' and regime in ['bullish', 'sideways']:
                confidence -= np.random.randint(5, 15)
        
        # Adjust based on strategy
        if strategy == 'momentum' and direction == 'long':
            confidence += np.random.randint(0, 10)
        elif strategy == 'mean_reversion':
            confidence += np.random.randint(-5, 15)
        
        # Ensure confidence is between 0-100
        confidence = max(0, min(100, confidence))
        
        # Generate risk/reward ratio if stop loss and take profit provided
        risk_reward_text = ""
        if stop_loss and take_profit and price:
            if direction == 'long':
                reward = take_profit - price
                risk = price - stop_loss
            else:  # short
                reward = price - take_profit
                risk = stop_loss - price
            
            if risk > 0:
                risk_reward = reward / risk
                risk_reward_text = f" The risk/reward ratio is {risk_reward:.2f}."
                
                # Adjust confidence based on risk/reward
                if risk_reward > 2:
                    confidence += np.random.randint(5, 15)
                elif risk_reward < 1:
                    confidence -= np.random.randint(5, 15)
                
                # Ensure confidence is between 0-100 after adjustment
                confidence = max(0, min(100, confidence))
        
        # Generate technical analysis
        technical_analyses = [
            f"The price is showing a potential {direction} setup for {strategy} strategy.",
            f"Technical indicators suggest a moderate {direction} bias for {symbol}.",
            f"Recent price action indicates a possible {direction} opportunity with {strategy} approach.",
            f"Chart patterns show a developing {direction} trend that aligns with {strategy} strategy."
        ]
        
        # Generate news sentiment
        news_sentiments = [
            f"Recent news for {symbol} has been generally neutral with slight positive bias.",
            f"News sentiment appears mixed but with some favorable reports for a {direction} position.",
            f"Media coverage shows balanced views with no significant impact expected.",
            f"News analysis indicates low coverage recently with minimal sentiment impact."
        ]
        
        # Generate recommendations
        if confidence >= 70:
            recommendations = [
                f"Strong {direction} opportunity that aligns well with {strategy} strategy.",
                f"High conviction {direction} setup with favorable risk/reward profile.",
                f"Recommending this {direction} trade with high confidence based on analysis."
            ]
        elif confidence >= 50:
            recommendations = [
                f"Moderate {direction} opportunity that fits {strategy} parameters.",
                f"Consider a {direction} position with defined risk management.",
                f"Acceptable {direction} setup with average conviction level."
            ]
        else:
            recommendations = [
                f"Low conviction {direction} opportunity; consider reducing position size.",
                f"Marginal {direction} setup; only enter if additional confirmation appears.",
                f"Weak {direction} signal that doesn't fully align with {strategy} criteria."
            ]
        
        # Generate risks
        risks = [
            f"Unexpected earnings announcement or company news",
            f"Broader market volatility affecting {symbol}",
            f"Sector rotation away from {symbol}'s industry",
            f"Key technical level breach invalidating the setup",
            f"Liquidity concerns during trade execution",
            f"Increased correlation with market indices"
        ]
        
        # Generate adjustments
        adjustments = [
            f"Consider tightening stop loss to reduce risk exposure",
            f"Implement a scaled entry approach instead of full position",
            f"Add a secondary confirmation indicator before entry",
            f"Wait for pullback to key support level for better entry",
            f"Consider options strategy instead for defined risk"
        ]
        
        # Randomly select items for response
        technical_analysis = np.random.choice(technical_analyses)
        news_sentiment = np.random.choice(news_sentiments)
        recommendation = np.random.choice(recommendations)
        
        selected_risks = list(np.random.choice(risks, size=min(3, len(risks)), replace=False))
        selected_adjustments = list(np.random.choice(adjustments, size=min(2, len(adjustments)), replace=False))
        
        return {
            'confidence_score': int(confidence),
            'technical_analysis': technical_analysis + risk_reward_text,
            'news_sentiment': news_sentiment,
            'trade_recommendation': recommendation,
            'key_risks': selected_risks,
            'suggested_adjustments': selected_adjustments
        }
    
    def fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch market data for a symbol from external sources.
        This is a placeholder - implement with your data provider.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Market data dictionary
        """
        logger.info(f"Fetch market data for {symbol}")
        # Placeholder implementation - replace with actual fetching logic
        return {"status": "ok", "symbol": symbol, "last_price": 100.0}
        
    def fetch_news_data(self, symbol: str, days: int = 3) -> List[Dict[str, Any]]:
        """
        Fetch recent news for a symbol from external sources.
        This is a placeholder - implement with your news provider.
        
        Args:
            symbol: Trading symbol
            days: Number of days to look back
            
        Returns:
            List of news article dictionaries
        """
        # This should be implemented with your preferred news provider
        logger.warning("fetch_news_data is a placeholder - implement with your news provider")
        return []
    
    def get_recent_evaluations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent trade evaluations.
        
        Args:
            limit: Maximum number of evaluations to return
            
        Returns:
            List of evaluation dictionaries
        """
        return self.evaluations[-limit:]
    
    def save_evaluations(self, output_path: str) -> None:
        """
        Save all evaluations to a file.
        
        Args:
            output_path: Path to save evaluations
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.evaluations, f, indent=2)
        
        logger.info(f"Saved {len(self.evaluations)} evaluations to {output_path}")


# Example implementation of market data fetcher
class NewsDataFetcher:
    """Helper class to fetch news data from various sources"""
    
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize the news data fetcher."""
        # Load configuration
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Set API key from config, parameter, or environment
        self.api_key = (
            api_key or 
            self.config.get('news_api_key') or 
            os.environ.get('NEWS_API_KEY')
        )
    
    def fetch_alpha_vantage_news(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch news from Alpha Vantage News API.
        
        Args:
            symbol: Trading symbol
            limit: Maximum number of articles to return
            
        Returns:
            List of news articles
        """
        if not self.api_key:
            logger.warning("No API key provided for Alpha Vantage")
            return []
        
        try:
            # Alpha Vantage News API endpoint
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "apikey": self.api_key,
                "limit": limit
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if "feed" not in data:
                logger.warning(f"No news data returned from Alpha Vantage for {symbol}")
                return []
            
            # Format articles
            articles = []
            for item in data["feed"][:limit]:
                article = {
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", ""),
                    "date": item.get("time_published", ""),
                    "url": item.get("url", ""),
                    "sentiment": item.get("overall_sentiment_score", 0)
                }
                articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news from Alpha Vantage: {str(e)}")
            return []
    
    def fetch_newsapi_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch news from NewsAPI.
        
        Args:
            symbol: Trading symbol
            days: Number of days to look back
            
        Returns:
            List of news articles
        """
        if not self.api_key:
            logger.warning("No API key provided for NewsAPI")
            return []
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # NewsAPI endpoint
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": symbol,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "language": "en",
                "sortBy": "publishedAt",
                "apiKey": self.api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get("status") != "ok" or "articles" not in data:
                logger.warning(f"No news data returned from NewsAPI for {symbol}")
                return []
            
            # Format articles
            articles = []
            for item in data["articles"]:
                article = {
                    "title": item.get("title", ""),
                    "summary": item.get("description", ""),
                    "content": item.get("content", ""),
                    "source": item.get("source", {}).get("name", ""),
                    "date": item.get("publishedAt", ""),
                    "url": item.get("url", "")
                }
                articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news from NewsAPI: {str(e)}")
            return []
    
    def generate_mock_news(self, symbol: str, count: int = 5) -> List[Dict[str, Any]]:
        """
        Generate mock news data for testing.
        
        Args:
            symbol: Trading symbol
            count: Number of articles to generate
            
        Returns:
            List of mock news articles
        """
        # Set seed for reproducibility
        np.random.seed(hash(symbol) % 10000)
        
        # Potential titles and content
        positive_titles = [
            f"{symbol} Exceeds Q3 Expectations",
            f"Analysts Upgrade {symbol} Citing Strong Growth",
            f"New Product Launch Drives {symbol} Stock Higher",
            f"{symbol} Announces Strategic Partnership",
            f"Market Leaders Praise {symbol}'s Innovation"
        ]
        
        negative_titles = [
            f"{symbol} Falls Short of Q3 Estimates",
            f"Analysts Downgrade {symbol} on Growth Concerns",
            f"Production Delays Impact {symbol}'s Outlook",
            f"{symbol} Faces Increased Competition",
            f"Market Uncertainty Weighs on {symbol}"
        ]
        
        neutral_titles = [
            f"{symbol} Reports In-Line Results",
            f"{symbol} Maintains Market Position",
            f"Industry Changes May Impact {symbol}",
            f"{symbol} CEO Comments on Market Trends",
            f"{symbol} Announces Board Changes"
        ]
        
        # Generate articles
        articles = []
        for i in range(count):
            # Determine sentiment
            sentiment = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.3, 0.3])
            
            # Select title based on sentiment
            if sentiment == 'positive':
                title = np.random.choice(positive_titles)
                sentiment_score = np.random.uniform(0.2, 0.9)
            elif sentiment == 'negative':
                title = np.random.choice(negative_titles)
                sentiment_score = np.random.uniform(-0.9, -0.2)
            else:
                title = np.random.choice(neutral_titles)
                sentiment_score = np.random.uniform(-0.1, 0.1)
            
            # Generate date
            days_ago = np.random.randint(0, 7)
            date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            # Create article
            article = {
                "title": title,
                "summary": f"Mock summary for {symbol} article with {sentiment} sentiment.",
                "source": np.random.choice(["MarketWatch", "Bloomberg", "Reuters", "CNBC", "WSJ"]),
                "date": date,
                "url": f"https://example.com/news/{symbol.lower()}/{i}",
                "sentiment": sentiment_score
            }
            
            articles.append(article)
        
        return articles 