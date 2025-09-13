#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlternativeDataConnector - Component to access and process alternative data sources
such as options flow, sentiment analysis, on-chain analytics, and dark pool activity.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import asyncio
import aiohttp
import requests
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

# Setup logging
logger = logging.getLogger("AlternativeDataConnector")

class AlternativeDataSource(ABC):
    """Abstract base class for alternative data sources."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the alternative data source.
        
        Args:
            name: Source name
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.api_key = self.config.get('api_key')
        self.api_secret = self.config.get('api_secret')
        self.base_url = self.config.get('base_url')
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour default
        self.cache = {}
        self.last_update = {}
        
        # Verify required configuration
        self._verify_config()
    
    def _verify_config(self) -> None:
        """Verify that required configuration is present."""
        required_config = self.get_required_config()
        missing = [key for key in required_config if key not in self.config]
        
        if missing:
            logger.warning(f"Missing required configuration for {self.name}: {', '.join(missing)}")
    
    @abstractmethod
    def get_required_config(self) -> List[str]:
        """
        Get list of required configuration keys.
        
        Returns:
            List of required configuration keys
        """
        pass
    
    @abstractmethod
    async def fetch_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch data for a symbol.
        
        Args:
            symbol: Symbol to fetch data for
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with fetched data
        """
        pass
    
    async def get_data(self, symbol: str, force_refresh: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Get data for a symbol, using cache if available.
        
        Args:
            symbol: Symbol to get data for
            force_refresh: Whether to force a refresh
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with data
        """
        cache_key = self._get_cache_key(symbol, kwargs)
        
        # Check if data is in cache and still valid
        if (not force_refresh and 
            self.cache_enabled and 
            cache_key in self.cache and 
            cache_key in self.last_update):
            
            # Check if cache is still valid
            elapsed = (datetime.now() - self.last_update[cache_key]).total_seconds()
            if elapsed < self.cache_ttl:
                logger.debug(f"Using cached data for {symbol} from {self.name}")
                return self.cache[cache_key]
        
        # Fetch fresh data
        try:
            data = await self.fetch_data(symbol, **kwargs)
            
            # Update cache
            if self.cache_enabled:
                self.cache[cache_key] = data
                self.last_update[cache_key] = datetime.now()
                
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from {self.name}: {str(e)}")
            
            # Return cached data if available, even if expired
            if cache_key in self.cache:
                logger.warning(f"Using expired cached data for {symbol} from {self.name}")
                return self.cache[cache_key]
                
            # Return empty data
            return {}
    
    def _get_cache_key(self, symbol: str, params: Dict[str, Any]) -> str:
        """
        Get cache key for a set of parameters.
        
        Args:
            symbol: Symbol
            params: Additional parameters
            
        Returns:
            Cache key string
        """
        # Create a key that uniquely identifies this request
        key_parts = [f"symbol={symbol}"]
        
        # Add sorted parameters
        for k, v in sorted(params.items()):
            # Convert lists/dicts to strings
            if isinstance(v, (list, dict)):
                v = json.dumps(v, sort_keys=True)
            key_parts.append(f"{k}={v}")
            
        return "|".join(key_parts)
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache = {}
        self.last_update = {}
        logger.debug(f"Cleared cache for {self.name}")


class OptionsFlowDataSource(AlternativeDataSource):
    """Data source for options flow analytics."""
    
    def get_required_config(self) -> List[str]:
        """Get required configuration keys."""
        return ['api_key', 'base_url']
    
    async def fetch_data(self, symbol: str, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         min_amount: Optional[float] = None) -> Dict[str, Any]:
        """
        Fetch options flow data.
        
        Args:
            symbol: Symbol to fetch data for
            start_date: Start date for data range
            end_date: End date for data range
            min_amount: Minimum order amount to include
            
        Returns:
            Dictionary with options flow data
        """
        # Default date range if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=7)
            
        # Build request parameters
        params = {
            'symbol': symbol,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'api_key': self.api_key
        }
        
        if min_amount is not None:
            params['min_amount'] = min_amount
        
        # Make API request
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/options/flow"
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Process response
                        if 'data' in result:
                            return self._process_options_data(result['data'], symbol)
                        else:
                            logger.warning(f"No data returned for {symbol} options flow")
                            return {}
                    else:
                        logger.error(f"Error fetching options flow data: {response.status}")
                        return {}
            except Exception as e:
                logger.error(f"Exception fetching options flow data: {str(e)}")
                return {}
    
    def _process_options_data(self, data: List[Dict[str, Any]], symbol: str) -> Dict[str, Any]:
        """
        Process options flow data.
        
        Args:
            data: Raw options flow data
            symbol: Symbol
            
        Returns:
            Processed options data
        """
        if not data:
            return {
                'symbol': symbol,
                'call_volume': 0,
                'put_volume': 0,
                'call_put_ratio': 0,
                'unusual_activity': [],
                'large_trades': []
            }
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(data)
        
        # Calculate aggregated metrics
        call_volume = df[df['option_type'] == 'call']['volume'].sum()
        put_volume = df[df['option_type'] == 'put']['volume'].sum()
        call_put_ratio = call_volume / put_volume if put_volume > 0 else float('inf')
        
        # Identify unusual activity
        unusual_activity = df[df['unusual'] == True].to_dict('records')
        
        # Identify large trades
        threshold = np.percentile(df['premium'], 95) if len(df) >= 20 else df['premium'].max() * 0.8
        large_trades = df[df['premium'] >= threshold].to_dict('records')
        
        # Calculate most active expiration dates
        expirations = df.groupby('expiration_date')['volume'].sum().sort_values(ascending=False)
        active_expirations = [
            {'date': date, 'volume': volume} 
            for date, volume in expirations.items()
        ]
        
        # Calculate most active strike prices
        strikes = df.groupby('strike_price')['volume'].sum().sort_values(ascending=False)
        active_strikes = [
            {'strike': strike, 'volume': volume} 
            for strike, volume in strikes.items()
        ]
        
        return {
            'symbol': symbol,
            'call_volume': int(call_volume),
            'put_volume': int(put_volume),
            'call_put_ratio': round(call_put_ratio, 2),
            'unusual_activity': unusual_activity[:10],  # Limit to top 10
            'large_trades': large_trades[:10],  # Limit to top 10
            'active_expirations': active_expirations[:5],  # Top 5 active expirations
            'active_strikes': active_strikes[:5],  # Top 5 active strikes
            'data_count': len(df),
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_sentiment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze options flow sentiment.
        
        Args:
            data: Options flow data
            
        Returns:
            Sentiment analysis
        """
        if not data or 'call_put_ratio' not in data:
            return {'sentiment': 'neutral', 'confidence': 0}
        
        # Analyze call/put ratio
        ratio = data['call_put_ratio']
        if ratio > 2.5:
            sentiment = 'very bullish'
            confidence = 0.9
        elif ratio > 1.5:
            sentiment = 'bullish'
            confidence = 0.7
        elif ratio > 1.2:
            sentiment = 'mildly bullish'
            confidence = 0.6
        elif ratio < 0.4:
            sentiment = 'very bearish'
            confidence = 0.9
        elif ratio < 0.7:
            sentiment = 'bearish'
            confidence = 0.7
        elif ratio < 0.9:
            sentiment = 'mildly bearish'
            confidence = 0.6
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        # Adjust based on unusual activity
        unusual_count = len(data.get('unusual_activity', []))
        if unusual_count > 0:
            unusual_calls = sum(1 for item in data.get('unusual_activity', []) if item.get('option_type') == 'call')
            unusual_puts = unusual_count - unusual_calls
            
            # Adjust sentiment if there's significant unusual activity
            if unusual_count >= 3:
                if unusual_calls > unusual_puts * 2:
                    sentiment = 'bullish'
                    confidence = 0.8
                elif unusual_puts > unusual_calls * 2:
                    sentiment = 'bearish'
                    confidence = 0.8
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'call_put_ratio': ratio,
            'unusual_activity_count': unusual_count
        }


class SentimentDataSource(AlternativeDataSource):
    """Data source for social media and news sentiment."""
    
    def get_required_config(self) -> List[str]:
        """Get required configuration keys."""
        return ['api_key', 'base_url']
    
    async def fetch_data(self, symbol: str, 
                         sources: Optional[List[str]] = None, 
                         lookback_days: int = 7) -> Dict[str, Any]:
        """
        Fetch sentiment data.
        
        Args:
            symbol: Symbol to fetch data for
            sources: List of sources to include
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with sentiment data
        """
        # Default sources if not provided
        if sources is None:
            sources = ['twitter', 'reddit', 'news']
        
        # Build request parameters
        params = {
            'symbol': symbol,
            'sources': ','.join(sources),
            'days': lookback_days,
            'api_key': self.api_key
        }
        
        # Make API request
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/sentiment"
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Process response
                        if 'data' in result:
                            return self._process_sentiment_data(result['data'], symbol)
                        else:
                            logger.warning(f"No sentiment data returned for {symbol}")
                            return {}
                    else:
                        logger.error(f"Error fetching sentiment data: {response.status}")
                        return {}
            except Exception as e:
                logger.error(f"Exception fetching sentiment data: {str(e)}")
                return {}
    
    def _process_sentiment_data(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """
        Process sentiment data.
        
        Args:
            data: Raw sentiment data
            symbol: Symbol
            
        Returns:
            Processed sentiment data
        """
        if not data:
            return {
                'symbol': symbol,
                'sentiment_score': 0,
                'sentiment': 'neutral',
                'volume': 0,
                'sources': {}
            }
        
        # Extract overall sentiment
        overall = data.get('overall', {})
        sentiment_score = overall.get('sentiment_score', 0)
        
        # Determine sentiment category
        if sentiment_score >= 0.7:
            sentiment = 'very positive'
        elif sentiment_score >= 0.3:
            sentiment = 'positive'
        elif sentiment_score >= 0.1:
            sentiment = 'slightly positive'
        elif sentiment_score <= -0.7:
            sentiment = 'very negative'
        elif sentiment_score <= -0.3:
            sentiment = 'negative'
        elif sentiment_score <= -0.1:
            sentiment = 'slightly negative'
        else:
            sentiment = 'neutral'
        
        # Process source-specific data
        sources_data = {}
        for source, source_data in data.get('sources', {}).items():
            sources_data[source] = {
                'sentiment_score': source_data.get('sentiment_score', 0),
                'volume': source_data.get('volume', 0),
                'positive': source_data.get('positive', 0),
                'negative': source_data.get('negative', 0),
                'neutral': source_data.get('neutral', 0)
            }
        
        # Get sentiment over time if available
        time_series = []
        for item in data.get('time_series', []):
            time_series.append({
                'date': item.get('date'),
                'sentiment_score': item.get('sentiment_score', 0),
                'volume': item.get('volume', 0)
            })
        
        return {
            'symbol': symbol,
            'sentiment_score': sentiment_score,
            'sentiment': sentiment,
            'volume': overall.get('volume', 0),
            'sources': sources_data,
            'time_series': time_series,
            'timestamp': datetime.now().isoformat()
        }


class OnChainDataSource(AlternativeDataSource):
    """Data source for blockchain on-chain analytics."""
    
    def get_required_config(self) -> List[str]:
        """Get required configuration keys."""
        return ['api_key', 'base_url']
    
    async def fetch_data(self, symbol: str, 
                         metrics: Optional[List[str]] = None,
                         days: int = 30) -> Dict[str, Any]:
        """
        Fetch on-chain data.
        
        Args:
            symbol: Symbol to fetch data for
            metrics: List of on-chain metrics to include
            days: Number of days to look back
            
        Returns:
            Dictionary with on-chain data
        """
        # Default metrics if not provided
        if metrics is None:
            metrics = ['active_addresses', 'transaction_count', 'transaction_volume', 'nvt_ratio']
        
        # Build request parameters
        params = {
            'asset': symbol,
            'metrics': ','.join(metrics),
            'days': days,
            'api_key': self.api_key
        }
        
        # Make API request
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/onchain/metrics"
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Process response
                        if 'data' in result:
                            return self._process_onchain_data(result['data'], symbol)
                        else:
                            logger.warning(f"No on-chain data returned for {symbol}")
                            return {}
                    else:
                        logger.error(f"Error fetching on-chain data: {response.status}")
                        return {}
            except Exception as e:
                logger.error(f"Exception fetching on-chain data: {str(e)}")
                return {}
    
    def _process_onchain_data(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """
        Process on-chain data.
        
        Args:
            data: Raw on-chain data
            symbol: Symbol
            
        Returns:
            Processed on-chain data
        """
        if not data:
            return {
                'symbol': symbol,
                'metrics': {},
                'time_series': [],
                'analysis': {}
            }
        
        # Process metrics
        metrics = {}
        for metric, value in data.get('metrics', {}).items():
            metrics[metric] = value
        
        # Process time series data
        time_series = []
        for entry in data.get('time_series', []):
            time_series.append({
                'date': entry.get('date'),
                'metrics': entry.get('metrics', {})
            })
        
        # Create derived indicators
        analysis = self._analyze_onchain_metrics(metrics, time_series)
        
        return {
            'symbol': symbol,
            'metrics': metrics,
            'time_series': time_series,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_onchain_metrics(self, metrics: Dict[str, Any], 
                                time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze on-chain metrics to derive indicators.
        
        Args:
            metrics: Current on-chain metrics
            time_series: Historical time series data
            
        Returns:
            Analysis of on-chain metrics
        """
        analysis = {}
        
        # Skip analysis if not enough data
        if not metrics or not time_series or len(time_series) < 7:
            return {
                'network_health': 'unknown',
                'adoption_trend': 'unknown',
                'hodl_trend': 'unknown'
            }
        
        # Create DataFrame from time series
        df_metrics = {}
        
        # First determine what metrics we have
        available_metrics = set()
        for ts in time_series:
            available_metrics.update(ts.get('metrics', {}).keys())
        
        # Initialize metric lists
        for metric in available_metrics:
            df_metrics[metric] = []
        
        df_metrics['date'] = []
        
        # Populate metric lists
        for ts in time_series:
            ts_metrics = ts.get('metrics', {})
            df_metrics['date'].append(ts.get('date'))
            for metric in available_metrics:
                df_metrics[metric].append(ts_metrics.get(metric, None))
        
        # Create DataFrame
        df = pd.DataFrame(df_metrics)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Calculate 7-day and 30-day moving averages for key metrics
        for metric in ['active_addresses', 'transaction_count', 'transaction_volume']:
            if metric in df.columns:
                df[f'{metric}_7d_ma'] = df[metric].rolling(7).mean()
                df[f'{metric}_30d_ma'] = df[metric].rolling(30).mean()
                
                # Calculate percent change from 7 days ago
                if len(df) >= 7:
                    current = df[metric].iloc[-1]
                    seven_days_ago = df[metric].iloc[-7]
                    pct_change_7d = ((current / seven_days_ago) - 1) * 100 if seven_days_ago else 0
                    analysis[f'{metric}_7d_pct_change'] = pct_change_7d
        
        # Analyze network health
        if 'active_addresses' in df.columns and 'transaction_count' in df.columns:
            # Calculate correlation between active addresses and transaction count
            corr = df['active_addresses'].corr(df['transaction_count'])
            
            # Calculate recent trend
            if 'active_addresses_7d_ma' in df.columns and len(df) >= 7:
                recent_trend = df['active_addresses_7d_ma'].iloc[-1] / df['active_addresses_7d_ma'].iloc[-7] - 1
                
                if corr > 0.7 and recent_trend > 0.05:
                    network_health = 'very good'
                elif corr > 0.5 and recent_trend > 0:
                    network_health = 'good'
                elif corr > 0.3 and recent_trend > -0.05:
                    network_health = 'moderate'
                elif corr > 0 and recent_trend > -0.1:
                    network_health = 'concerning'
                else:
                    network_health = 'poor'
                    
                analysis['network_health'] = network_health
                analysis['network_metrics_correlation'] = corr
                analysis['network_growth_trend'] = recent_trend
        
        # Analyze adoption trend
        if 'active_addresses' in df.columns and len(df) >= 30:
            # Compare 7-day MA to 30-day MA
            if 'active_addresses_7d_ma' in df.columns and 'active_addresses_30d_ma' in df.columns:
                ratio = df['active_addresses_7d_ma'].iloc[-1] / df['active_addresses_30d_ma'].iloc[-1]
                
                if ratio > 1.2:
                    adoption_trend = 'accelerating'
                elif ratio > 1.05:
                    adoption_trend = 'growing'
                elif ratio > 0.95:
                    adoption_trend = 'stable'
                elif ratio > 0.8:
                    adoption_trend = 'declining'
                else:
                    adoption_trend = 'rapidly declining'
                    
                analysis['adoption_trend'] = adoption_trend
                analysis['adoption_ma_ratio'] = ratio
        
        # Analyze holding trend (HODL wave)
        if 'supply_pct_held_1y_plus' in df.columns:
            # Check long-term holding percentage trend
            if len(df) >= 30:
                current = df['supply_pct_held_1y_plus'].iloc[-1]
                month_ago = df['supply_pct_held_1y_plus'].iloc[-30]
                
                hold_change = current - month_ago
                
                if hold_change > 5:
                    hodl_trend = 'strong accumulation'
                elif hold_change > 2:
                    hodl_trend = 'accumulation'
                elif hold_change > -2:
                    hodl_trend = 'neutral'
                elif hold_change > -5:
                    hodl_trend = 'distribution'
                else:
                    hodl_trend = 'strong distribution'
                    
                analysis['hodl_trend'] = hodl_trend
                analysis['hodl_30d_change'] = hold_change
        
        return analysis


class DarkPoolDataSource(AlternativeDataSource):
    """Data source for dark pool and OTC trading activity."""
    
    def get_required_config(self) -> List[str]:
        """Get required configuration keys."""
        return ['api_key', 'base_url']
    
    async def fetch_data(self, symbol: str, days: int = 10) -> Dict[str, Any]:
        """
        Fetch dark pool data.
        
        Args:
            symbol: Symbol to fetch data for
            days: Number of days to look back
            
        Returns:
            Dictionary with dark pool data
        """
        # Build request parameters
        params = {
            'symbol': symbol,
            'days': days,
            'api_key': self.api_key
        }
        
        # Make API request
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/darkpool/data"
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Process response
                        if 'data' in result:
                            return self._process_darkpool_data(result['data'], symbol)
                        else:
                            logger.warning(f"No dark pool data returned for {symbol}")
                            return {}
                    else:
                        logger.error(f"Error fetching dark pool data: {response.status}")
                        return {}
            except Exception as e:
                logger.error(f"Exception fetching dark pool data: {str(e)}")
                return {}
    
    def _process_darkpool_data(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """
        Process dark pool data.
        
        Args:
            data: Raw dark pool data
            symbol: Symbol
            
        Returns:
            Processed dark pool data
        """
        if not data:
            return {
                'symbol': symbol,
                'total_volume': 0,
                'dark_pool_percentage': 0,
                'price_levels': [],
                'time_series': []
            }
        
        # Extract summary data
        total_volume = data.get('total_volume', 0)
        dark_pool_volume = data.get('dark_pool_volume', 0)
        dark_pool_percentage = (dark_pool_volume / total_volume * 100) if total_volume > 0 else 0
        
        # Process price levels
        price_levels = []
        for level in data.get('price_levels', []):
            price_levels.append({
                'price': level.get('price'),
                'volume': level.get('volume'),
                'timestamp': level.get('timestamp')
            })
        
        # Process time series
        time_series = []
        for item in data.get('time_series', []):
            time_series.append({
                'date': item.get('date'),
                'dark_pool_volume': item.get('dark_pool_volume', 0),
                'total_volume': item.get('total_volume', 0),
                'percentage': item.get('percentage', 0)
            })
        
        return {
            'symbol': symbol,
            'total_volume': total_volume,
            'dark_pool_volume': dark_pool_volume,
            'dark_pool_percentage': dark_pool_percentage,
            'price_levels': sorted(price_levels, key=lambda x: x.get('volume', 0), reverse=True),
            'time_series': time_series,
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_dark_pool_levels(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze dark pool price levels to identify support/resistance.
        
        Args:
            data: Dark pool data
            
        Returns:
            List of significant price levels
        """
        if not data or 'price_levels' not in data:
            return []
        
        price_levels = data.get('price_levels', [])
        if not price_levels:
            return []
        
        # Convert to DataFrame
        df = pd.DataFrame(price_levels)
        
        # Calculate total volume
        total_volume = df['volume'].sum()
        
        # Calculate percentage of total volume for each level
        df['volume_pct'] = df['volume'] / total_volume * 100
        
        # Find significant levels (>5% of total volume)
        significant = df[df['volume_pct'] > 5].copy()
        
        # Sort by volume (highest first)
        significant = significant.sort_values('volume', ascending=False)
        
        return significant.to_dict('records')


class AlternativeDataConnector:
    """
    Main connector for alternative data sources that aggregates and processes data
    from multiple alternative sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AlternativeDataConnector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.data_sources = {}
        
        # Initialize data sources based on configuration
        self._initialize_data_sources()
    
    def _initialize_data_sources(self) -> None:
        """Initialize data sources from configuration."""
        # Check for options flow configuration
        if 'options_flow' in self.config:
            self.data_sources['options_flow'] = OptionsFlowDataSource(
                name="OptionsFlow",
                config=self.config.get('options_flow')
            )
        
        # Check for sentiment configuration
        if 'sentiment' in self.config:
            self.data_sources['sentiment'] = SentimentDataSource(
                name="Sentiment",
                config=self.config.get('sentiment')
            )
        
        # Check for on-chain configuration
        if 'onchain' in self.config:
            self.data_sources['onchain'] = OnChainDataSource(
                name="OnChain",
                config=self.config.get('onchain')
            )
        
        # Check for dark pool configuration
        if 'dark_pool' in self.config:
            self.data_sources['dark_pool'] = DarkPoolDataSource(
                name="DarkPool",
                config=self.config.get('dark_pool')
            )
        
        logger.info(f"Initialized {len(self.data_sources)} alternative data sources")
    
    def add_data_source(self, source_id: str, source: AlternativeDataSource) -> None:
        """
        Add a data source to the connector.
        
        Args:
            source_id: Identifier for the source
            source: Data source instance
        """
        self.data_sources[source_id] = source
        logger.info(f"Added data source: {source.name}")
    
    def remove_data_source(self, source_id: str) -> bool:
        """
        Remove a data source from the connector.
        
        Args:
            source_id: Identifier for the source
            
        Returns:
            bool: True if source was removed, False otherwise
        """
        if source_id in self.data_sources:
            del self.data_sources[source_id]
            logger.info(f"Removed data source: {source_id}")
            return True
        return False
    
    async def get_data(self, symbol: str, sources: Optional[List[str]] = None,
                      force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get data from multiple alternative sources.
        
        Args:
            symbol: Symbol to get data for
            sources: List of sources to include (if None, use all available)
            force_refresh: Whether to force a refresh
            
        Returns:
            Dictionary with data from all sources
        """
        # Determine which sources to use
        if sources is None:
            sources = list(self.data_sources.keys())
        else:
            # Filter to only include available sources
            sources = [s for s in sources if s in self.data_sources]
        
        # Fetch data from all sources concurrently
        tasks = {}
        for source_id in sources:
            source = self.data_sources.get(source_id)
            if source:
                tasks[source_id] = asyncio.create_task(source.get_data(symbol, force_refresh=force_refresh))
        
        # Wait for all tasks to complete
        results = {}
        for source_id, task in tasks.items():
            try:
                results[source_id] = await task
            except Exception as e:
                logger.error(f"Error getting data from {source_id}: {str(e)}")
                results[source_id] = {}
        
        # Process combined results
        return self._process_combined_data(results, symbol)
    
    def _process_combined_data(self, results: Dict[str, Dict[str, Any]], symbol: str) -> Dict[str, Any]:
        """
        Process combined data from multiple sources.
        
        Args:
            results: Results from multiple sources
            symbol: Symbol
            
        Returns:
            Processed combined data
        """
        combined = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': list(results.keys()),
            'data': results
        }
        
        # Add sentiment analysis if sentiment data is available
        if 'sentiment' in results:
            combined['sentiment_analysis'] = results['sentiment']
        
        # Add options flow analysis if options data is available
        if 'options_flow' in results and 'options_flow' in self.data_sources:
            options_source = self.data_sources['options_flow']
            combined['options_analysis'] = options_source.analyze_sentiment(results['options_flow'])
        
        # Add dark pool analysis if dark pool data is available
        if 'dark_pool' in results and 'dark_pool' in self.data_sources:
            dark_pool_source = self.data_sources['dark_pool']
            significant_levels = dark_pool_source.analyze_dark_pool_levels(results['dark_pool'])
            combined['significant_dark_pool_levels'] = significant_levels
        
        # Calculate composite score
        combined['market_signals'] = self._calculate_composite_signals(results, symbol)
        
        return combined
    
    def _calculate_composite_signals(self, results: Dict[str, Dict[str, Any]], 
                                    symbol: str) -> Dict[str, Any]:
        """
        Calculate composite signals from multiple data sources.
        
        Args:
            results: Results from multiple sources
            symbol: Symbol
            
        Returns:
            Composite signals
        """
        signals = {
            'overall_sentiment': 'neutral',
            'sentiment_score': 0,
            'confidence': 0,
            'significant_factors': []
        }
        
        sentiment_factors = []
        sentiment_scores = []
        confidences = []
        
        # Process sentiment data
        if 'sentiment' in results and results['sentiment']:
            sentiment_data = results['sentiment']
            if 'sentiment_score' in sentiment_data:
                score = sentiment_data['sentiment_score']
                sentiment_scores.append(score)
                confidences.append(0.7)  # Base confidence for sentiment data
                
                sentiment_factors.append({
                    'source': 'social_sentiment',
                    'score': score,
                    'sentiment': sentiment_data.get('sentiment', 'neutral')
                })
        
        # Process options flow
        if 'options_flow' in results and 'options_analysis' in self.data_sources:
            options_analysis = self.data_sources['options_flow'].analyze_sentiment(results['options_flow'])
            
            if 'sentiment' in options_analysis:
                options_sentiment = options_analysis['sentiment']
                
                # Convert sentiment to score
                if options_sentiment == 'very bullish':
                    score = 0.9
                elif options_sentiment == 'bullish':
                    score = 0.7
                elif options_sentiment == 'mildly bullish':
                    score = 0.3
                elif options_sentiment == 'very bearish':
                    score = -0.9
                elif options_sentiment == 'bearish':
                    score = -0.7
                elif options_sentiment == 'mildly bearish':
                    score = -0.3
                else:
                    score = 0
                
                sentiment_scores.append(score)
                confidences.append(options_analysis.get('confidence', 0.5))
                
                sentiment_factors.append({
                    'source': 'options_flow',
                    'score': score,
                    'sentiment': options_sentiment
                })
        
        # Process dark pool data
        if 'dark_pool' in results and results['dark_pool']:
            dark_pool_data = results['dark_pool']
            
            # Check for significant dark pool activity
            if 'dark_pool_percentage' in dark_pool_data:
                dark_pool_pct = dark_pool_data['dark_pool_percentage']
                
                # Higher dark pool percentage often indicates institutional activity
                if dark_pool_pct > 50:
                    factor = {
                        'source': 'dark_pool',
                        'significance': 'very high',
                        'percentage': dark_pool_pct
                    }
                    signals['significant_factors'].append(factor)
        
        # Process on-chain data for crypto
        if 'onchain' in results and results['onchain']:
            onchain_data = results['onchain']
            
            if 'analysis' in onchain_data:
                analysis = onchain_data['analysis']
                
                # Check network health
                if 'network_health' in analysis:
                    network_health = analysis['network_health']
                    
                    # Convert network health to score
                    if network_health == 'very good':
                        score = 0.8
                    elif network_health == 'good':
                        score = 0.6
                    elif network_health == 'moderate':
                        score = 0.2
                    elif network_health == 'concerning':
                        score = -0.3
                    elif network_health == 'poor':
                        score = -0.7
                    else:
                        score = 0
                    
                    sentiment_scores.append(score)
                    confidences.append(0.6)  # Base confidence for on-chain data
                    
                    sentiment_factors.append({
                        'source': 'onchain_health',
                        'score': score,
                        'sentiment': network_health
                    })
                
                # Check adoption trend
                if 'adoption_trend' in analysis:
                    adoption_trend = analysis['adoption_trend']
                    
                    # Convert adoption trend to score
                    if adoption_trend == 'accelerating':
                        score = 0.8
                    elif adoption_trend == 'growing':
                        score = 0.5
                    elif adoption_trend == 'stable':
                        score = 0.1
                    elif adoption_trend == 'declining':
                        score = -0.5
                    elif adoption_trend == 'rapidly declining':
                        score = -0.8
                    else:
                        score = 0
                    
                    sentiment_scores.append(score)
                    confidences.append(0.5)  # Base confidence for adoption data
                    
                    sentiment_factors.append({
                        'source': 'onchain_adoption',
                        'score': score,
                        'sentiment': adoption_trend
                    })
        
        # Calculate weighted average sentiment score
        if sentiment_scores and confidences:
            total_confidence = sum(confidences)
            if total_confidence > 0:
                weighted_score = sum(s * c for s, c in zip(sentiment_scores, confidences)) / total_confidence
                signals['sentiment_score'] = round(weighted_score, 2)
                signals['confidence'] = round(sum(confidences) / len(confidences), 2)
                
                # Determine overall sentiment
                if weighted_score >= 0.7:
                    overall_sentiment = 'very bullish'
                elif weighted_score >= 0.3:
                    overall_sentiment = 'bullish'
                elif weighted_score >= 0.1:
                    overall_sentiment = 'slightly bullish'
                elif weighted_score <= -0.7:
                    overall_sentiment = 'very bearish'
                elif weighted_score <= -0.3:
                    overall_sentiment = 'bearish'
                elif weighted_score <= -0.1:
                    overall_sentiment = 'slightly bearish'
                else:
                    overall_sentiment = 'neutral'
                
                signals['overall_sentiment'] = overall_sentiment
        
        # Add sentiment factors
        signals['sentiment_factors'] = sentiment_factors
        
        return signals
    
    def clear_cache(self) -> None:
        """Clear cache for all data sources."""
        for source in self.data_sources.values():
            source.clear_cache()
        logger.info("Cleared cache for all data sources")


# Example usage
async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example configuration
    config = {
        'options_flow': {
            'api_key': 'your_api_key',
            'base_url': 'https://api.example.com',
            'cache_enabled': True,
            'cache_ttl': 3600  # 1 hour
        },
        'sentiment': {
            'api_key': 'your_api_key',
            'base_url': 'https://api.example.com',
            'cache_enabled': True,
            'cache_ttl': 3600  # 1 hour
        }
    }
    
    # Create connector
    connector = AlternativeDataConnector(config)
    
    # Get data for a symbol
    data = await connector.get_data('AAPL', sources=['options_flow', 'sentiment'])
    
    # Print results
    print(f"Data for AAPL from {len(data['sources'])} sources:")
    print(f"Overall sentiment: {data['market_signals']['overall_sentiment']}")
    print(f"Sentiment score: {data['market_signals']['sentiment_score']}")
    print(f"Confidence: {data['market_signals']['confidence']}")
    
    if 'sentiment_factors' in data['market_signals']:
        print("\nSentiment factors:")
        for factor in data['market_signals']['sentiment_factors']:
            print(f"- {factor['source']}: {factor['sentiment']} (score: {factor['score']})")
    
    if 'significant_factors' in data['market_signals']:
        print("\nSignificant factors:")
        for factor in data['market_signals']['significant_factors']:
            print(f"- {factor['source']}: {factor.get('significance')}")


if __name__ == "__main__":
    asyncio.run(main()) 