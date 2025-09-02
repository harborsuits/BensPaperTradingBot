"""
Options Data Integration Module

This module bridges the OptionsMarketData class with the MultiAssetAdapter, providing:
1. Seamless integration of options data into the existing trading system
2. Automatic handling of data gaps and fallbacks
3. Connection to the OptionsPerformanceAnalyzer for metrics tracking
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from datetime import datetime, timedelta

from trading_bot.data.options_market_data import OptionsMarketData, DataSourcePriority
from trading_bot.utils.enhanced_trade_executor import EnhancedTradeExecutor
from trading_bot.performance.options_performance_analyzer import OptionsPerformanceAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

class OptionsDataIntegration:
    """
    Integrates options market data with the MultiAssetAdapter and provides
    a unified interface for accessing options data across the trading system.
    """
    
    def __init__(
        self,
        multi_asset_adapter,
        config_path: Optional[str] = None,
        options_market_data: Optional[OptionsMarketData] = None,
        performance_analyzer: Optional[OptionsPerformanceAnalyzer] = None,
        cache_dir: str = 'data/options_cache',
        enable_metrics_tracking: bool = True
    ):
        """
        Initialize the options data integration.
        
        Args:
            multi_asset_adapter: Instance of MultiAssetAdapter
            config_path: Path to configuration file
            options_market_data: Optional pre-configured OptionsMarketData instance
            performance_analyzer: Optional OptionsPerformanceAnalyzer instance
            cache_dir: Directory for caching options data
            enable_metrics_tracking: Whether to track metrics about data quality and latency
        """
        self.multi_asset_adapter = multi_asset_adapter
        self.config_path = config_path
        self.cache_dir = cache_dir
        self.enable_metrics_tracking = enable_metrics_tracking
        
        # Initialize options market data if not provided
        if options_market_data:
            self.options_market_data = options_market_data
        else:
            self.options_market_data = OptionsMarketData(
                config_path=config_path,
                cache_dir=cache_dir
            )
        
        # Connect to performance analyzer if provided
        self.performance_analyzer = performance_analyzer
        
        # Track metrics about data quality
        self.metrics = {
            'requests': 0,
            'cache_hits': 0,
            'fallbacks_used': 0,
            'errors': 0,
            'avg_latency_ms': 0,
            'data_gaps_filled': 0
        }
        
        # Register additional data sources from adapter configuration
        self._register_adapter_data_sources()
        
        logger.info("Initialized OptionsDataIntegration")
    
    def _register_adapter_data_sources(self):
        """Register any data sources from the multi-asset adapter configuration"""
        try:
            # Get broker connections from adapter
            brokers = getattr(self.multi_asset_adapter, 'brokers', {})
            
            # Check if there's a dedicated options broker
            if 'options' in brokers and brokers['options']:
                broker_config = self._extract_broker_config(brokers['options'])
                if broker_config:
                    self.options_market_data.register_data_source(
                        broker_config.get('name', 'adapter_options'),
                        broker_config,
                        DataSourcePriority.PRIMARY
                    )
            
            # Check for multi-asset brokers that might support options
            for asset_class, broker in brokers.items():
                if asset_class != 'options' and broker:
                    # Check if this broker supports options
                    if self._broker_supports_options(broker):
                        broker_config = self._extract_broker_config(broker)
                        if broker_config:
                            self.options_market_data.register_data_source(
                                broker_config.get('name', f'adapter_{asset_class}'),
                                broker_config,
                                DataSourcePriority.SECONDARY
                            )
        except Exception as e:
            logger.error(f"Error registering adapter data sources: {str(e)}")
    
    def _extract_broker_config(self, broker) -> Dict[str, Any]:
        """Extract configuration from broker object for data source registration"""
        try:
            config = {}
            
            # Try to get broker name
            if hasattr(broker, 'name'):
                config['name'] = broker.name
            elif hasattr(broker, 'broker_name'):
                config['name'] = broker.broker_name
            
            # Try to get API credentials
            if hasattr(broker, 'api_key'):
                config['api_key'] = broker.api_key
            
            if hasattr(broker, 'api_secret'):
                config['secret_key'] = broker.api_secret
            
            # Get endpoint information if available
            if hasattr(broker, 'base_url'):
                config['base_url'] = broker.base_url
            
            return config
        except Exception as e:
            logger.error(f"Error extracting broker config: {str(e)}")
            return {}
    
    def _broker_supports_options(self, broker) -> bool:
        """Check if a broker supports options trading/data"""
        # This is a simple heuristic check - could be improved with more specific logic
        if not broker:
            return False
        
        # Check broker name if available
        if hasattr(broker, 'name'):
            name = broker.name.lower()
            if any(provider in name for provider in ['tradier', 'thinkorswim', 'tdameritrade', 'ibkr', 'interactive']):
                return True
        
        # Check for options-specific methods
        options_methods = [
            'get_options_chain',
            'get_option_chain',
            'get_options',
            'place_option_order',
            'get_option_positions'
        ]
        
        for method in options_methods:
            if hasattr(broker, method) and callable(getattr(broker, method)):
                return True
        
        return False
    
    def get_options_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Get options chain for a symbol, integrated with the MultiAssetAdapter.
        
        Args:
            symbol: Underlying symbol
            expiration: Option expiration date (format: YYYY-MM-DD)
            force_refresh: Whether to bypass cache
            
        Returns:
            Options chain data
        """
        # Normalize symbol using the adapter's method if available
        if hasattr(self.multi_asset_adapter, 'normalize_symbol'):
            normalized_symbol = self.multi_asset_adapter.normalize_symbol(symbol, 'options')
        else:
            normalized_symbol = symbol
        
        start_time = datetime.now()
        self.metrics['requests'] += 1
        
        try:
            # Get options chain from dedicated options market data
            options_data = self.options_market_data.get_options_chain(
                normalized_symbol, expiration, force_refresh
            )
            
            # Calculate and update metrics
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics('success', latency)
            
            # Check if this is using cached data
            if not force_refresh and 'timestamp' in options_data:
                cache_time = datetime.fromisoformat(options_data['timestamp']) \
                    if isinstance(options_data['timestamp'], str) else options_data['timestamp']
                if (datetime.now() - cache_time).total_seconds() < self.options_market_data.cache_expiry_minutes * 60:
                    self.metrics['cache_hits'] += 1
            
            # Track in performance analyzer if available
            if self.performance_analyzer and self.enable_metrics_tracking:
                self._track_data_metrics(normalized_symbol, options_data, 'options_chain')
            
            return options_data
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {str(e)}")
            self._update_metrics('error', (datetime.now() - start_time).total_seconds() * 1000)
            
            # Return empty chain as fallback
            return {
                'options': {'option': []},
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_iv_surface(
        self,
        symbol: str,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get implied volatility surface for a symbol.
        
        Args:
            symbol: Underlying symbol
            date: Date for historical surface (None for current)
            
        Returns:
            IV surface data
        """
        # Normalize symbol using the adapter's method if available
        if hasattr(self.multi_asset_adapter, 'normalize_symbol'):
            normalized_symbol = self.multi_asset_adapter.normalize_symbol(symbol, 'options')
        else:
            normalized_symbol = symbol
        
        start_time = datetime.now()
        self.metrics['requests'] += 1
        
        try:
            # Get IV surface from options market data
            surface_data = self.options_market_data.get_iv_surface(normalized_symbol, date)
            
            # Calculate and update metrics
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics('success', latency)
            
            # Track in performance analyzer if available
            if self.performance_analyzer and self.enable_metrics_tracking:
                self._track_data_metrics(normalized_symbol, surface_data, 'iv_surface')
            
            return surface_data
        except Exception as e:
            logger.error(f"Error getting IV surface for {symbol}: {str(e)}")
            self._update_metrics('error', (datetime.now() - start_time).total_seconds() * 1000)
            
            # Return empty surface as fallback
            return {
                'underlying': symbol,
                'error': str(e),
                'surface': [],
                'timestamp': datetime.now().isoformat()
            }
    
    def get_iv_history(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = 'daily',
        fill_gaps: bool = True
    ) -> pd.DataFrame:
        """
        Get implied volatility history with gap filling.
        
        Args:
            symbol: Underlying symbol
            start_date: Start date (format: YYYY-MM-DD)
            end_date: End date (format: YYYY-MM-DD)
            period: Period ('daily', 'weekly', 'monthly')
            fill_gaps: Whether to fill gaps in the data
            
        Returns:
            DataFrame with IV history
        """
        # Normalize symbol using the adapter's method if available
        if hasattr(self.multi_asset_adapter, 'normalize_symbol'):
            normalized_symbol = self.multi_asset_adapter.normalize_symbol(symbol, 'options')
        else:
            normalized_symbol = symbol
        
        start_time = datetime.now()
        self.metrics['requests'] += 1
        
        try:
            # Get IV history from options market data
            iv_data = self.options_market_data.get_iv_history(
                normalized_symbol, start_date, end_date, period
            )
            
            # Fill gaps if requested
            if fill_gaps and not iv_data.empty:
                original_len = len(iv_data)
                iv_data = self.options_market_data.fill_data_gaps(iv_data)
                if len(iv_data) > original_len:
                    self.metrics['data_gaps_filled'] += 1
            
            # Calculate and update metrics
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics('success', latency)
            
            # Track in performance analyzer if available
            if self.performance_analyzer and self.enable_metrics_tracking and not iv_data.empty:
                self._track_iv_history_metrics(normalized_symbol, iv_data)
            
            return iv_data
        except Exception as e:
            logger.error(f"Error getting IV history for {symbol}: {str(e)}")
            self._update_metrics('error', (datetime.now() - start_time).total_seconds() * 1000)
            
            # Return empty DataFrame as fallback
            return pd.DataFrame()
    
    def _update_metrics(self, result: str, latency_ms: float):
        """Update metrics tracking"""
        if not self.enable_metrics_tracking:
            return
        
        if result == 'error':
            self.metrics['errors'] += 1
            self.metrics['fallbacks_used'] += 1
        
        # Update average latency
        total_requests = self.metrics['requests']
        if total_requests > 1:
            # Rolling average
            self.metrics['avg_latency_ms'] = (
                self.metrics['avg_latency_ms'] * (total_requests - 1) + latency_ms
            ) / total_requests
        else:
            self.metrics['avg_latency_ms'] = latency_ms
    
    def _track_data_metrics(self, symbol: str, data: Dict[str, Any], data_type: str):
        """Track data metrics in performance analyzer"""
        if not self.performance_analyzer:
            return
        
        try:
            # Basic data quality metrics
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'data_type': data_type,
                'request_success': 'error' not in data,
                'latency_ms': self.metrics['avg_latency_ms']
            }
            
            # Add data-specific metrics
            if data_type == 'options_chain':
                options = data.get('options', {}).get('option', [])
                metrics.update({
                    'option_count': len(options),
                    'has_calls': any(opt.get('type') == 'call' for opt in options),
                    'has_puts': any(opt.get('type') == 'put' for opt in options),
                    'has_greeks': any('greeks' in opt for opt in options),
                })
            elif data_type == 'iv_surface':
                surface_points = data.get('surface', [])
                metrics.update({
                    'surface_point_count': len(surface_points),
                    'expiration_count': len(data.get('expirations', {})),
                })
            
            # Track in performance analyzer
            if hasattr(self.performance_analyzer, 'track_data_metrics'):
                self.performance_analyzer.track_data_metrics(metrics)
        except Exception as e:
            logger.error(f"Error tracking data metrics: {str(e)}")
    
    def _track_iv_history_metrics(self, symbol: str, iv_data: pd.DataFrame):
        """Track IV history metrics in performance analyzer"""
        if not self.performance_analyzer:
            return
        
        try:
            # Extract basic statistics
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'data_type': 'iv_history',
                'data_points': len(iv_data),
                'start_date': iv_data['date'].iloc[0] if not iv_data.empty else None,
                'end_date': iv_data['date'].iloc[-1] if not iv_data.empty else None,
                'avg_iv': iv_data['iv'].mean() if 'iv' in iv_data.columns and not iv_data.empty else None,
                'max_iv': iv_data['iv'].max() if 'iv' in iv_data.columns and not iv_data.empty else None,
                'min_iv': iv_data['iv'].min() if 'iv' in iv_data.columns and not iv_data.empty else None,
                'current_iv_percentile': iv_data['iv_percentile'].iloc[-1] if 'iv_percentile' in iv_data.columns and not iv_data.empty else None,
            }
            
            # Track in performance analyzer
            if hasattr(self.performance_analyzer, 'track_data_metrics'):
                self.performance_analyzer.track_data_metrics(metrics)
        except Exception as e:
            logger.error(f"Error tracking IV history metrics: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics about data quality and performance"""
        return self.metrics
    
    def reset_metrics(self):
        """Reset metrics counters"""
        self.metrics = {
            'requests': 0,
            'cache_hits': 0,
            'fallbacks_used': 0,
            'errors': 0,
            'avg_latency_ms': 0,
            'data_gaps_filled': 0
        }
        
    def integrate_with_adapter(self):
        """
        Integrate options data capabilities with the MultiAssetAdapter.
        This method extends the adapter with options-specific data methods.
        """
        # Skip if already integrated
        if hasattr(self.multi_asset_adapter, '_options_data_integrated'):
            return
        
        # Store reference to self in the adapter
        self.multi_asset_adapter._options_data_integration = self
        
        # Add options data methods to adapter
        def get_options_chain(adapter, symbol, expiration=None, force_refresh=False):
            return adapter._options_data_integration.get_options_chain(
                symbol, expiration, force_refresh
            )
        
        def get_iv_surface(adapter, symbol, date=None):
            return adapter._options_data_integration.get_iv_surface(symbol, date)
        
        def get_iv_history(adapter, symbol, start_date=None, end_date=None, period='daily', fill_gaps=True):
            return adapter._options_data_integration.get_iv_history(
                symbol, start_date, end_date, period, fill_gaps
            )
        
        # Attach methods to adapter
        import types
        setattr(self.multi_asset_adapter, 'get_options_chain', 
                types.MethodType(get_options_chain, self.multi_asset_adapter))
        
        setattr(self.multi_asset_adapter, 'get_iv_surface',
                types.MethodType(get_iv_surface, self.multi_asset_adapter))
        
        setattr(self.multi_asset_adapter, 'get_iv_history',
                types.MethodType(get_iv_history, self.multi_asset_adapter))
        
        # Mark as integrated
        self.multi_asset_adapter._options_data_integrated = True
        
        logger.info("Successfully integrated options data methods with MultiAssetAdapter") 