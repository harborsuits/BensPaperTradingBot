#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alternative Data Integration Module for BensBot

This module implements the infrastructure for integrating alternative data sources
into BensBot's trading signals and decision-making process. It provides connectors
for various alternative data sources and methods to enrich traditional market data.

Alternative data sources include:
- Social media sentiment
- News sentiment analysis
- Macroeconomic indicators
- Web traffic and search trends
- Satellite imagery data (when available)
- ESG (Environmental, Social, Governance) metrics
"""

import pandas as pd
import numpy as np
import logging
import datetime
import requests
import json
from typing import Dict, List, Union, Tuple, Optional, Callable
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)


class AlternativeDataSource(ABC):
    """
    Abstract base class for all alternative data sources.
    Each concrete implementation must provide data fetching and processing methods.
    """
    
    def __init__(self, name: str, refresh_interval_minutes: int = 60):
        """
        Initialize the data source.
        
        Args:
            name: Name of the data source
            refresh_interval_minutes: How often to refresh the data (in minutes)
        """
        self.name = name
        self.refresh_interval_minutes = refresh_interval_minutes
        self.last_refresh_time = None
        self.raw_data = None
        self.processed_data = None
        
        logger.info(f"Initialized {name} alternative data source")
    
    @abstractmethod
    def fetch_data(self, **kwargs) -> pd.DataFrame:
        """
        Fetch raw data from the source.
        
        Args:
            **kwargs: Source-specific parameters
            
        Returns:
            DataFrame containing the raw data
        """
        pass
    
    @abstractmethod
    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw data into a format usable by trading strategies.
        
        Args:
            raw_data: Raw data from the source
            
        Returns:
            Processed data ready for use in trading strategies
        """
        pass
    
    def get_data(self, force_refresh: bool = False, **kwargs) -> pd.DataFrame:
        """
        Get data from the source, refreshing if needed.
        
        Args:
            force_refresh: Whether to force a refresh regardless of the time since last refresh
            **kwargs: Source-specific parameters
            
        Returns:
            Processed data ready for use in trading strategies
        """
        current_time = datetime.datetime.now()
        
        # Check if we need to refresh
        needs_refresh = (
            force_refresh or 
            self.last_refresh_time is None or
            (current_time - self.last_refresh_time).total_seconds() / 60 >= self.refresh_interval_minutes
        )
        
        if needs_refresh:
            logger.info(f"Refreshing data from {self.name}")
            try:
                self.raw_data = self.fetch_data(**kwargs)
                self.processed_data = self.process_data(self.raw_data)
                self.last_refresh_time = current_time
                logger.info(f"Successfully refreshed data from {self.name}")
            except Exception as e:
                logger.error(f"Failed to refresh data from {self.name}: {str(e)}")
                # If we have no data at all, raise the exception
                if self.processed_data is None:
                    raise
                # Otherwise, just log the error and continue with stale data
        
        return self.processed_data


class SocialMediaSentiment(AlternativeDataSource):
    """
    Social media sentiment analysis for various assets.
    Tracks mentions, sentiment, and engagement across platforms.
    """
    
    def __init__(self, api_key: str = None, platforms: List[str] = None, 
               refresh_interval_minutes: int = 30):
        """
        Initialize social media sentiment data source.
        
        Args:
            api_key: API key for the sentiment provider
            platforms: List of platforms to track (twitter, reddit, etc.)
            refresh_interval_minutes: How often to refresh the data
        """
        super().__init__("Social Media Sentiment", refresh_interval_minutes)
        self.api_key = api_key
        self.platforms = platforms or ["twitter", "reddit", "stocktwits"]
        
        logger.info(f"Initialized social media sentiment tracking for {', '.join(self.platforms)}")
    
    def fetch_data(self, symbols: List[str], lookback_days: int = 7, **kwargs) -> pd.DataFrame:
        """
        Fetch social media sentiment data for the specified symbols.
        
        Args:
            symbols: List of ticker symbols to fetch data for
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with sentiment data
        """
        # In a real implementation, this would call an API service
        # For now, we generate mock data
        
        logger.info(f"Fetching social media sentiment for {len(symbols)} symbols over {lookback_days} days")
        
        # Create date range
        end_date = datetime.datetime.now().date()
        start_date = end_date - datetime.timedelta(days=lookback_days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate mock data
        data = []
        for symbol in symbols:
            for date in dates:
                for platform in self.platforms:
                    # Generate random sentiment scores
                    sentiment_score = np.random.normal(0, 0.6)  # Mean 0, std 0.6
                    sentiment_score = max(min(sentiment_score, 1), -1)  # Clip to [-1, 1]
                    
                    # Generate random mention counts and engagement
                    mention_count = int(np.random.exponential(20 if symbol[0] < 'M' else 10))
                    engagement_score = np.random.exponential(mention_count / 10)
                    
                    data.append({
                        'date': date,
                        'symbol': symbol,
                        'platform': platform,
                        'sentiment_score': sentiment_score,
                        'mention_count': mention_count,
                        'engagement_score': engagement_score
                    })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        logger.debug(f"Generated {len(df)} social media sentiment records")
        
        return df
    
    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw sentiment data into a format usable by trading strategies.
        
        Args:
            raw_data: Raw sentiment data
            
        Returns:
            Processed sentiment data with daily aggregated metrics
        """
        if raw_data.empty:
            return pd.DataFrame()
        
        logger.info("Processing social media sentiment data")
        
        # Group by date and symbol, aggregating metrics
        aggregated = raw_data.groupby(['date', 'symbol']).agg({
            'sentiment_score': ['mean', 'std'],
            'mention_count': 'sum',
            'engagement_score': 'sum'
        }).reset_index()
        
        # Flatten multi-level columns
        aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]
        
        # Calculate additional metrics
        aggregated['sentiment_signal'] = aggregated['sentiment_score_mean'] * np.log1p(aggregated['mention_count'])
        aggregated['sentiment_intensity'] = aggregated['sentiment_score_std'] * aggregated['engagement_score'] / 100
        
        # Create separate platform metrics if needed
        if not raw_data.empty and 'platform' in raw_data.columns:
            # Get per-platform sentiment for each symbol
            platform_sentiment = raw_data.pivot_table(
                index=['date', 'symbol'],
                columns='platform',
                values='sentiment_score',
                aggfunc='mean'
            ).reset_index()
            
            # Rename platform columns
            platform_sentiment.columns.name = None
            platform_cols = [col for col in platform_sentiment.columns if col not in ['date', 'symbol']]
            for col in platform_cols:
                platform_sentiment.rename(columns={col: f'{col}_sentiment'}, inplace=True)
            
            # Merge with aggregated data
            aggregated = pd.merge(aggregated, platform_sentiment, on=['date', 'symbol'], how='left')
        
        return aggregated


class NewsSentimentAnalyzer(AlternativeDataSource):
    """
    News sentiment analysis for various assets and sectors.
    Tracks news volume, sentiment, and relevance.
    """
    
    def __init__(self, api_key: str = None, sources: List[str] = None,
               refresh_interval_minutes: int = 60):
        """
        Initialize news sentiment analyzer.
        
        Args:
            api_key: API key for the news API
            sources: List of news sources to track
            refresh_interval_minutes: How often to refresh the data
        """
        super().__init__("News Sentiment", refresh_interval_minutes)
        self.api_key = api_key
        self.sources = sources or ["bloomberg", "reuters", "wsj", "cnbc", "ft"]
        
        logger.info(f"Initialized news sentiment analyzer with {len(self.sources)} sources")
    
    def fetch_data(self, symbols: List[str], lookback_days: int = 3, **kwargs) -> pd.DataFrame:
        """
        Fetch news sentiment data for the specified symbols.
        
        Args:
            symbols: List of ticker symbols to fetch data for
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with news sentiment data
        """
        # In a real implementation, this would call a news API
        # For now, we generate mock data
        
        logger.info(f"Fetching news sentiment for {len(symbols)} symbols over {lookback_days} days")
        
        # Create date range with hourly frequency
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=lookback_days)
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Generate mock data
        data = []
        for symbol in symbols:
            # Generate some news stories for each symbol
            num_stories = np.random.poisson(lookback_days * 2)  # Average 2 stories per day
            
            for _ in range(num_stories):
                # Random timestamp within the lookback period
                timestamp = dates[np.random.randint(0, len(dates))]
                
                # Random source
                source = np.random.choice(self.sources)
                
                # Generate random sentiment and relevance scores
                sentiment_score = np.random.normal(0, 0.5)  # Mean 0, std 0.5
                sentiment_score = max(min(sentiment_score, 1), -1)  # Clip to [-1, 1]
                
                relevance_score = np.random.beta(2, 2)  # Between 0 and 1, centered around 0.5
                
                # Generate mock headline
                headline = f"Mock headline about {symbol} from {source}"
                
                data.append({
                    'timestamp': timestamp,
                    'date': timestamp.date(),
                    'symbol': symbol,
                    'source': source,
                    'headline': headline,
                    'sentiment_score': sentiment_score,
                    'relevance_score': relevance_score
                })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        logger.debug(f"Generated {len(df)} news sentiment records")
        
        return df
    
    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw news sentiment data into a format usable by trading strategies.
        
        Args:
            raw_data: Raw news sentiment data
            
        Returns:
            Processed news sentiment data with daily aggregated metrics
        """
        if raw_data.empty:
            return pd.DataFrame()
        
        logger.info("Processing news sentiment data")
        
        # Group by date and symbol, aggregating metrics
        aggregated = raw_data.groupby(['date', 'symbol']).agg({
            'sentiment_score': ['mean', 'count', 'std'],
            'relevance_score': 'mean'
        }).reset_index()
        
        # Flatten multi-level columns
        aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]
        
        # Rename some columns for clarity
        aggregated.rename(columns={
            'sentiment_score_count': 'news_count',
            'relevance_score_mean': 'relevance_score'
        }, inplace=True)
        
        # Calculate weighted sentiment score (by relevance)
        relevance_weighted = raw_data.copy()
        relevance_weighted['weighted_sentiment'] = relevance_weighted['sentiment_score'] * relevance_weighted['relevance_score']
        
        weighted_agg = relevance_weighted.groupby(['date', 'symbol']).agg({
            'weighted_sentiment': 'sum',
            'relevance_score': 'sum'
        }).reset_index()
        
        weighted_agg['relevance_weighted_sentiment'] = weighted_agg['weighted_sentiment'] / weighted_agg['relevance_score']
        
        # Merge into main aggregated dataframe
        aggregated = pd.merge(
            aggregated,
            weighted_agg[['date', 'symbol', 'relevance_weighted_sentiment']],
            on=['date', 'symbol'],
            how='left'
        )
        
        # Add a news momentum signal
        # Filter to last 3 days
        if not raw_data.empty and 'date' in raw_data.columns:
            latest_date = raw_data['date'].max()
            three_days_ago = latest_date - datetime.timedelta(days=3)
            recent_news = raw_data[raw_data['date'] >= three_days_ago]
            
            if not recent_news.empty:
                # Group by symbol and day distance from latest
                recent_news['days_ago'] = (latest_date - recent_news['date']).dt.days
                
                # Weight by recency (more recent news has higher weight)
                recent_news['recency_weight'] = 1 / (recent_news['days_ago'] + 1)
                recent_news['weighted_sent'] = recent_news['sentiment_score'] * recent_news['recency_weight'] * recent_news['relevance_score']
                
                momentum = recent_news.groupby('symbol').agg({
                    'weighted_sent': 'sum',
                    'recency_weight': 'sum',
                    'relevance_score': 'sum'
                })
                
                momentum['news_momentum'] = momentum['weighted_sent'] / momentum['recency_weight']
                momentum = momentum.reset_index()[['symbol', 'news_momentum']]
                
                # Merge momentum signal
                aggregated = pd.merge(
                    aggregated,
                    momentum,
                    on='symbol',
                    how='left'
                )
        
        return aggregated


class MacroeconomicIndicators(AlternativeDataSource):
    """
    Macroeconomic indicator data for market analysis.
    Includes GDP, inflation, employment, interest rates, and other macro factors.
    """
    
    def __init__(self, api_key: str = None, refresh_interval_minutes: int = 1440):
        """
        Initialize macroeconomic indicator data source.
        
        Args:
            api_key: API key for the data provider
            refresh_interval_minutes: How often to refresh the data (default: daily)
        """
        super().__init__("Macroeconomic Indicators", refresh_interval_minutes)
        self.api_key = api_key
        
        # Define indicator categories
        self.indicator_categories = {
            'growth': ['gdp', 'industrial_production', 'retail_sales'],
            'inflation': ['cpi', 'ppi', 'pce'],
            'employment': ['nonfarm_payrolls', 'unemployment_rate', 'jobless_claims'],
            'housing': ['housing_starts', 'home_sales', 'building_permits'],
            'monetary': ['fed_funds_rate', 'treasury_yield_10y', 'treasury_yield_2y'],
            'sentiment': ['consumer_confidence', 'business_confidence', 'pmi_manufacturing']
        }
        
        logger.info(f"Initialized macroeconomic indicators tracking")
    
    def fetch_data(self, countries: List[str] = None, lookback_days: int = 180, **kwargs) -> pd.DataFrame:
        """
        Fetch macroeconomic indicator data.
        
        Args:
            countries: List of countries to fetch data for (ISO codes)
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with macroeconomic data
        """
        # Default to major economies if not specified
        countries = countries or ['US', 'EU', 'UK', 'JP', 'CN']
        
        logger.info(f"Fetching macroeconomic data for {len(countries)} countries over {lookback_days} days")
        
        # Create date range
        end_date = datetime.datetime.now().date()
        start_date = end_date - datetime.timedelta(days=lookback_days)
        
        # Generate mock data
        data = []
        
        # For each country and indicator category
        for country in countries:
            for category, indicators in self.indicator_categories.items():
                for indicator in indicators:
                    # Determine frequency based on indicator
                    if indicator in ['gdp', 'housing_starts', 'home_sales', 'building_permits']:
                        freq = 'Q'  # Quarterly
                    elif indicator in ['cpi', 'ppi', 'pce', 'industrial_production', 'retail_sales', 
                                    'nonfarm_payrolls', 'consumer_confidence', 'business_confidence', 
                                    'pmi_manufacturing']:
                        freq = 'M'  # Monthly
                    elif indicator in ['jobless_claims']:
                        freq = 'W'  # Weekly
                    else:
                        freq = 'D'  # Daily
                    
                    # Create date range with appropriate frequency
                    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
                    
                    # Base value and trend for this indicator
                    if indicator in ['fed_funds_rate', 'treasury_yield_10y', 'treasury_yield_2y']:
                        base_value = np.random.uniform(1, 5)
                        trend = np.random.normal(0, 0.1)
                    elif indicator in ['unemployment_rate']:
                        base_value = np.random.uniform(3, 8)
                        trend = np.random.normal(0, 0.1)
                    elif indicator in ['gdp', 'industrial_production', 'retail_sales']:
                        base_value = np.random.uniform(1, 4)
                        trend = np.random.normal(0, 0.2)
                    elif indicator in ['cpi', 'ppi', 'pce']:
                        base_value = np.random.uniform(1, 3)
                        trend = np.random.normal(0, 0.1)
                    else:
                        base_value = np.random.uniform(50, 100)
                        trend = np.random.normal(0, 2)
                    
                    # Generate values with trend and some noise
                    for i, date in enumerate(dates):
                        value = base_value + trend * i + np.random.normal(0, base_value * 0.05)
                        
                        # Previous period for calculating change
                        prev_value = (base_value + trend * (i-1) + np.random.normal(0, base_value * 0.05) 
                                    if i > 0 else base_value)
                        
                        # Calculate period-over-period change
                        change = ((value - prev_value) / prev_value) if prev_value != 0 else 0
                        
                        data.append({
                            'date': date.date(),
                            'country': country,
                            'category': category,
                            'indicator': indicator,
                            'value': value,
                            'change': change,
                            'surprise': np.random.normal(0, 0.2)  # Random surprise vs consensus
                        })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        logger.debug(f"Generated {len(df)} macroeconomic data records")
        
        return df
    
    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw macroeconomic data into a format usable by trading strategies.
        
        Args:
            raw_data: Raw macroeconomic data
            
        Returns:
            Processed macroeconomic data
        """
        if raw_data.empty:
            return pd.DataFrame()
        
        logger.info("Processing macroeconomic data")
        
        # Create pivot table with indicators as columns
        pivot = raw_data.pivot_table(
            index=['date', 'country'],
            columns=['indicator'],
            values=['value', 'change', 'surprise']
        )
        
        # Flatten multi-level columns
        pivot.columns = ['_'.join(col) for col in pivot.columns.values]
        pivot = pivot.reset_index()
        
        # Calculate spreads for relevant indicators
        if not raw_data.empty and 'indicator' in raw_data.columns:
            # Get latest data for each country-indicator combination
            latest_data = raw_data.sort_values('date').groupby(['country', 'indicator']).last().reset_index()
            
            # Pivot to get indicators as columns for calculating spreads
            latest_pivot = latest_data.pivot(index='country', columns='indicator', values='value')
            
            # Calculate relevant spreads (e.g., 10y-2y Treasury spread)
            if 'treasury_yield_10y' in latest_pivot.columns and 'treasury_yield_2y' in latest_pivot.columns:
                latest_pivot['treasury_spread_10y_2y'] = latest_pivot['treasury_yield_10y'] - latest_pivot['treasury_yield_2y']
            
            # Calculate indicator composites
            # Growth composite
            growth_indicators = [ind for ind in self.indicator_categories['growth'] if ind in latest_pivot.columns]
            if growth_indicators:
                latest_pivot['growth_composite'] = latest_pivot[growth_indicators].mean(axis=1)
            
            # Inflation composite
            inflation_indicators = [ind for ind in self.indicator_categories['inflation'] if ind in latest_pivot.columns]
            if inflation_indicators:
                latest_pivot['inflation_composite'] = latest_pivot[inflation_indicators].mean(axis=1)
            
            # Reshape for merging
            latest_pivot = latest_pivot.reset_index()
            
            # Get only the calculated columns (spreads and composites)
            calc_columns = [col for col in latest_pivot.columns 
                         if col not in raw_data['indicator'].unique() and col != 'country']
            
            if calc_columns:
                latest_calcs = latest_pivot[['country'] + calc_columns]
                
                # Merge back to pivot
                latest_dates = pivot.groupby('country')['date'].max().reset_index()
                latest_with_dates = pd.merge(latest_dates, latest_calcs, on='country')
                
                # Now merge to full pivot
                for calc_col in calc_columns:
                    pivot = pd.merge(
                        pivot,
                        latest_with_dates[['country', 'date', calc_col]],
                        on=['country', 'date'],
                        how='left'
                    )
        
        return pivot


class AlternativeDataIntegrator:
    """
    Main class for integrating alternative data sources with traditional market data.
    Provides a unified interface for retrieving and combining data from multiple sources.
    """
    
    def __init__(self):
        """
        Initialize the alternative data integrator.
        """
        self.data_sources = {}
        logger.info("Initialized Alternative Data Integrator")
    
    def add_data_source(self, source_name: str, source: AlternativeDataSource):
        """
        Add a data source to the integrator.
        
        Args:
            source_name: Unique name for the data source
            source: AlternativeDataSource instance
        """
        self.data_sources[source_name] = source
        logger.info(f"Added data source: {source_name}")
    
    def get_data(self, source_name: str, **kwargs) -> pd.DataFrame:
        """
        Get data from a specific source.
        
        Args:
            source_name: Name of the data source
            **kwargs: Source-specific parameters
            
        Returns:
            DataFrame with data from the source
        """
        if source_name not in self.data_sources:
            raise ValueError(f"Data source '{source_name}' not found")
        
        return self.data_sources[source_name].get_data(**kwargs)
    
    def combine_data(self, market_data: pd.DataFrame, 
                  alt_data_sources: List[str] = None,
                  **kwargs) -> pd.DataFrame:
        """
        Combine market data with alternative data.
        
        Args:
            market_data: Traditional market data (OHLCV)
            alt_data_sources: List of alternative data sources to combine
            **kwargs: Source-specific parameters
            
        Returns:
            Combined dataframe with market and alternative data
        """
        if alt_data_sources is None:
            alt_data_sources = list(self.data_sources.keys())
        
        combined_data = market_data.copy()
        
        # Extract symbols from market data
        if 'symbol' in market_data.columns:
            symbols = market_data['symbol'].unique().tolist()
        else:
            # If no symbol column, assume a single symbol
            symbols = kwargs.get('symbols', [])
        
        # Combine each data source
        for source_name in alt_data_sources:
            if source_name not in self.data_sources:
                logger.warning(f"Data source '{source_name}' not found, skipping")
                continue
            
            try:
                # Get data for this source
                source_kwargs = kwargs.get(source_name, {})
                source_kwargs['symbols'] = symbols
                
                source_data = self.get_data(source_name, **source_kwargs)
                
                if not source_data.empty:
                    # Merge with combined data
                    if 'symbol' in source_data.columns and 'symbol' in combined_data.columns:
                        # If both have symbol column, merge on date and symbol
                        combined_data = pd.merge(
                            combined_data,
                            source_data,
                            on=['date', 'symbol'],
                            how='left'
                        )
                    elif 'date' in source_data.columns and 'date' in combined_data.columns:
                        # If only date column is common, merge on date
                        combined_data = pd.merge(
                            combined_data,
                            source_data,
                            on='date',
                            how='left'
                        )
                    else:
                        logger.warning(f"Cannot merge data from {source_name}, no common keys")
            
            except Exception as e:
                logger.error(f"Error combining data from {source_name}: {str(e)}")
        
        return combined_data
    
    def generate_features(self, combined_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading features from combined market and alternative data.
        
        Args:
            combined_data: Combined market and alternative data
            
        Returns:
            DataFrame with additional features for trading
        """
        data = combined_data.copy()
        
        # Check which alternative data columns are available
        columns = data.columns.tolist()
        
        # Generate features based on available columns
        
        # Social media features
        if 'sentiment_signal' in columns:
            # Smoothed sentiment
            data['sentiment_ma5'] = data.groupby('symbol')['sentiment_signal'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            
            # Sentiment momentum (change in sentiment)
            data['sentiment_momentum'] = data.groupby('symbol')['sentiment_ma5'].transform(
                lambda x: x.diff()
            )
        
        # News features
        if 'news_momentum' in columns:
            # Extreme news signal (high absolute sentiment with high news count)
            if 'sentiment_score_mean' in columns and 'news_count' in columns:
                data['extreme_news_signal'] = (
                    data['sentiment_score_mean'].abs() * 
                    np.log1p(data['news_count'])
                )
        
        # Macro features
        # Treasury spread signal (if available)
        if 'treasury_spread_10y_2y' in columns:
            # Negative spread is a recession indicator
            data['recession_signal'] = (data['treasury_spread_10y_2y'] < 0).astype(int)
        
        # Combined alternative data strength
        alt_cols = []
        if 'sentiment_signal' in columns:
            alt_cols.append('sentiment_signal')
        if 'news_momentum' in columns:
            alt_cols.append('news_momentum')
        
        if alt_cols:
            # Normalize and combine signals
            for col in alt_cols:
                # Min-max scaling within symbol groups
                data[f'{col}_norm'] = data.groupby('symbol')[col].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
                )
            
            # Combined alternative signal (average of normalized signals)
            norm_cols = [f'{col}_norm' for col in alt_cols]
            data['alt_data_signal'] = data[norm_cols].mean(axis=1)
        
        return data


def create_alternative_data_integrator() -> AlternativeDataIntegrator:
    """
    Factory function to create and configure an AlternativeDataIntegrator with common data sources.
    
    Returns:
        Configured AlternativeDataIntegrator instance
    """
    integrator = AlternativeDataIntegrator()
    
    # Add social media sentiment source
    social_sentiment = SocialMediaSentiment(refresh_interval_minutes=30)
    integrator.add_data_source('social_sentiment', social_sentiment)
    
    # Add news sentiment source
    news_sentiment = NewsSentimentAnalyzer(refresh_interval_minutes=60)
    integrator.add_data_source('news_sentiment', news_sentiment)
    
    # Add macroeconomic indicators source
    macro_indicators = MacroeconomicIndicators(refresh_interval_minutes=1440)
    integrator.add_data_source('macro_indicators', macro_indicators)
    
    return integrator
