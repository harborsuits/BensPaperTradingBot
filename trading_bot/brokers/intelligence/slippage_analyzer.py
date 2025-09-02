#!/usr/bin/env python3
"""
Slippage Analyzer for Broker Intelligence

Analyzes slippage metrics from order executions and provides feedback
to the broker intelligence system for scoring and routing decisions.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

from trading_bot.core.events import SlippageMetric
from trading_bot.event_system.event_bus import EventBus


class SlippageAnalyzer:
    """
    Analyzes execution slippage metrics to improve broker scoring and routing
    
    This component:
    1. Listens for SlippageMetric events from order executions
    2. Calculates rolling statistics on slippage by broker, asset class, and symbol
    3. Detects abnormal slippage patterns that might indicate broker issues
    4. Feeds slippage insights into broker intelligence for score adjustments
    """
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize the slippage analyzer
        
        Args:
            event_bus: Event bus to subscribe to slippage metrics
        """
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        
        # Store recent slippage metrics
        self.recent_metrics = {
            # Global metrics
            'all': [],
            
            # By broker
            'broker': defaultdict(list),
            
            # By asset class
            'asset_class': defaultdict(list),
            
            # By broker and asset class
            'broker_asset': defaultdict(list),
            
            # By symbol
            'symbol': defaultdict(list),
            
            # By broker and symbol
            'broker_symbol': defaultdict(list)
        }
        
        # Rolling statistics
        self.rolling_stats = {
            # By broker
            'broker': {},
            
            # By broker and asset class
            'broker_asset': {},
            
            # By broker and symbol
            'broker_symbol': {}
        }
        
        # Maximum age for metrics in memory (older metrics are pruned)
        self.max_age = timedelta(hours=24)
        
        # Subscribe to events
        self._subscribe_to_events()
    
    def _subscribe_to_events(self):
        """Subscribe to slippage metric events"""
        self.event_bus.on(SlippageMetric, self._on_slippage_metric)
    
    def _on_slippage_metric(self, event: SlippageMetric):
        """
        Handle slippage metric event
        
        Args:
            event: SlippageMetric event
        """
        try:
            # Extract data
            broker = event.broker
            symbol = event.symbol
            asset_class = event.asset_class
            slippage_bps = event.slippage_bps
            
            # Create metric object
            metric = {
                'broker': broker,
                'symbol': symbol,
                'asset_class': asset_class,
                'side': event.side,
                'expected_price': event.expected_price,
                'fill_price': event.fill_price,
                'slippage_amount': event.slippage_amount,
                'slippage_bps': slippage_bps,
                'order_id': event.order_id,
                'trade_id': event.trade_id,
                'timestamp': event.timestamp
            }
            
            # Add to appropriate lists
            self.recent_metrics['all'].append(metric)
            self.recent_metrics['broker'][broker].append(metric)
            self.recent_metrics['asset_class'][asset_class].append(metric)
            self.recent_metrics['broker_asset'][f"{broker}_{asset_class}"].append(metric)
            self.recent_metrics['symbol'][symbol].append(metric)
            self.recent_metrics['broker_symbol'][f"{broker}_{symbol}"].append(metric)
            
            # Update statistics
            self._update_statistics(broker, asset_class, symbol)
            
            # Prune old metrics
            self._prune_old_metrics()
            
            # Check for abnormal slippage
            self._check_abnormal_slippage(metric)
            
            self.logger.debug(f"Processed slippage metric: {broker}, {symbol}, {slippage_bps} bps")
            
        except Exception as e:
            self.logger.error(f"Error processing slippage metric: {str(e)}")
    
    def _update_statistics(self, broker: str, asset_class: str, symbol: str):
        """
        Update rolling statistics for a broker, asset class, and symbol
        
        Args:
            broker: Broker ID
            asset_class: Asset class
            symbol: Symbol
        """
        # Update broker statistics
        self._calculate_stats('broker', broker)
        
        # Update broker-asset statistics
        self._calculate_stats('broker_asset', f"{broker}_{asset_class}")
        
        # Update broker-symbol statistics
        self._calculate_stats('broker_symbol', f"{broker}_{symbol}")
    
    def _calculate_stats(self, category: str, key: str):
        """
        Calculate statistics for a category and key
        
        Args:
            category: Category ('broker', 'broker_asset', 'broker_symbol')
            key: Key within the category
        """
        metrics = self.recent_metrics[category][key]
        if not metrics:
            return
        
        # Calculate statistics
        slippage_values = [m['slippage_bps'] for m in metrics]
        
        stats = {
            'count': len(slippage_values),
            'mean': np.mean(slippage_values),
            'median': np.median(slippage_values),
            'std': np.std(slippage_values),
            'min': np.min(slippage_values),
            'max': np.max(slippage_values),
            'last_updated': datetime.now(),
            
            # Calculate percentiles
            'percentiles': {
                '10': np.percentile(slippage_values, 10),
                '25': np.percentile(slippage_values, 25),
                '75': np.percentile(slippage_values, 75),
                '90': np.percentile(slippage_values, 90)
            }
        }
        
        # Store stats
        self.rolling_stats[category][key] = stats
    
    def _prune_old_metrics(self):
        """Prune metrics older than max_age"""
        now = datetime.now()
        cutoff = now - self.max_age
        
        # Helper function to filter metrics
        def filter_recent(metrics_list):
            return [m for m in metrics_list if m['timestamp'] > cutoff]
        
        # Prune global metrics
        self.recent_metrics['all'] = filter_recent(self.recent_metrics['all'])
        
        # Prune broker metrics
        for broker in list(self.recent_metrics['broker'].keys()):
            self.recent_metrics['broker'][broker] = filter_recent(self.recent_metrics['broker'][broker])
        
        # Prune asset class metrics
        for asset_class in list(self.recent_metrics['asset_class'].keys()):
            self.recent_metrics['asset_class'][asset_class] = filter_recent(self.recent_metrics['asset_class'][asset_class])
        
        # Prune broker-asset metrics
        for key in list(self.recent_metrics['broker_asset'].keys()):
            self.recent_metrics['broker_asset'][key] = filter_recent(self.recent_metrics['broker_asset'][key])
        
        # Prune symbol metrics
        for symbol in list(self.recent_metrics['symbol'].keys()):
            self.recent_metrics['symbol'][symbol] = filter_recent(self.recent_metrics['symbol'][symbol])
        
        # Prune broker-symbol metrics
        for key in list(self.recent_metrics['broker_symbol'].keys()):
            self.recent_metrics['broker_symbol'][key] = filter_recent(self.recent_metrics['broker_symbol'][key])
    
    def _check_abnormal_slippage(self, metric: Dict[str, Any]):
        """
        Check for abnormal slippage
        
        Args:
            metric: Slippage metric
        """
        broker = metric['broker']
        symbol = metric['symbol']
        asset_class = metric['asset_class']
        slippage_bps = metric['slippage_bps']
        
        # Check broker statistics
        broker_stats = self.rolling_stats.get('broker', {}).get(broker)
        if broker_stats and broker_stats['count'] >= 10:
            # Check if slippage is abnormally high
            if slippage_bps > broker_stats['mean'] + 3 * broker_stats['std']:
                self._emit_abnormal_slippage_alert(
                    metric=metric,
                    category='broker',
                    stats=broker_stats,
                    z_score=(slippage_bps - broker_stats['mean']) / broker_stats['std']
                )
        
        # Check broker-asset statistics
        key = f"{broker}_{asset_class}"
        broker_asset_stats = self.rolling_stats.get('broker_asset', {}).get(key)
        if broker_asset_stats and broker_asset_stats['count'] >= 5:
            # Check if slippage is abnormally high
            if slippage_bps > broker_asset_stats['mean'] + 3 * broker_asset_stats['std']:
                self._emit_abnormal_slippage_alert(
                    metric=metric,
                    category='broker_asset',
                    stats=broker_asset_stats,
                    z_score=(slippage_bps - broker_asset_stats['mean']) / broker_asset_stats['std']
                )
        
        # Check broker-symbol statistics
        key = f"{broker}_{symbol}"
        broker_symbol_stats = self.rolling_stats.get('broker_symbol', {}).get(key)
        if broker_symbol_stats and broker_symbol_stats['count'] >= 3:
            # Check if slippage is abnormally high
            if slippage_bps > broker_symbol_stats['mean'] + 3 * broker_symbol_stats['std']:
                self._emit_abnormal_slippage_alert(
                    metric=metric,
                    category='broker_symbol',
                    stats=broker_symbol_stats,
                    z_score=(slippage_bps - broker_symbol_stats['mean']) / broker_symbol_stats['std']
                )
    
    def _emit_abnormal_slippage_alert(
        self, 
        metric: Dict[str, Any], 
        category: str, 
        stats: Dict[str, Any],
        z_score: float
    ):
        """
        Emit an alert for abnormal slippage
        
        Args:
            metric: Slippage metric
            category: Category ('broker', 'broker_asset', 'broker_symbol')
            stats: Statistics for the category
            z_score: Z-score of the abnormal slippage
        """
        broker = metric['broker']
        symbol = metric['symbol']
        asset_class = metric['asset_class']
        slippage_bps = metric['slippage_bps']
        
        # Log the alert
        self.logger.warning(
            f"ABNORMAL SLIPPAGE: {broker}, {symbol}, {slippage_bps} bps "
            f"(z-score: {z_score:.2f}, mean: {stats['mean']:.2f}, std: {stats['std']:.2f})"
        )
        
        # Create alert details
        details = {
            'broker': broker,
            'symbol': symbol,
            'asset_class': asset_class,
            'slippage_bps': slippage_bps,
            'mean_slippage': stats['mean'],
            'std_slippage': stats['std'],
            'z_score': z_score,
            'percentile_90': stats['percentiles']['90'],
            'category': category,
            'order_id': metric['order_id'],
            'trade_id': metric['trade_id'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Emit broker metric event for the broker intelligence system
        from trading_bot.core.events import BrokerMetric
        
        event = BrokerMetric(
            broker_id=broker,
            metric_type="slippage_anomaly",
            value=slippage_bps,
            unit="bps",
            operation="execution",
            asset_class=asset_class,
            metadata=details
        )
        
        self.event_bus.emit(event)
    
    def get_broker_slippage_stats(self, broker_id: str) -> Dict[str, Any]:
        """
        Get slippage statistics for a broker
        
        Args:
            broker_id: Broker ID
            
        Returns:
            Slippage statistics
        """
        return self.rolling_stats.get('broker', {}).get(broker_id, {})
    
    def get_broker_asset_slippage_stats(self, broker_id: str, asset_class: str) -> Dict[str, Any]:
        """
        Get slippage statistics for a broker and asset class
        
        Args:
            broker_id: Broker ID
            asset_class: Asset class
            
        Returns:
            Slippage statistics
        """
        key = f"{broker_id}_{asset_class}"
        return self.rolling_stats.get('broker_asset', {}).get(key, {})
    
    def get_broker_symbol_slippage_stats(self, broker_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get slippage statistics for a broker and symbol
        
        Args:
            broker_id: Broker ID
            symbol: Symbol
            
        Returns:
            Slippage statistics
        """
        key = f"{broker_id}_{symbol}"
        return self.rolling_stats.get('broker_symbol', {}).get(key, {})
    
    def get_all_broker_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get slippage statistics for all brokers
        
        Returns:
            Dictionary mapping broker IDs to slippage statistics
        """
        return self.rolling_stats.get('broker', {})


# Integration with broker intelligence system
def integrate_slippage_analysis_with_broker_intelligence(
    event_bus: EventBus,
    broker_advisor
) -> SlippageAnalyzer:
    """
    Integrate slippage analysis with the broker intelligence system
    
    Args:
        event_bus: Event bus
        broker_advisor: BrokerAdvisor instance
        
    Returns:
        SlippageAnalyzer instance
    """
    # Create slippage analyzer
    analyzer = SlippageAnalyzer(event_bus)
    
    # Register metric handler with broker advisor
    def handle_slippage_metric(broker_metric):
        if broker_metric.metric_type == "slippage_anomaly":
            # Extract data
            broker_id = broker_metric.broker_id
            slippage_bps = broker_metric.value
            z_score = broker_metric.metadata.get('z_score', 0)
            
            # Adjust broker scores based on slippage
            score_adjustment = min(max(-z_score * 2, -20), 0)  # Cap at -20 points
            
            broker_advisor.adjust_broker_score(
                broker_id=broker_id,
                score_adjustment=score_adjustment,
                category="execution_quality",
                reason=f"Abnormal slippage: {slippage_bps:.2f} bps (z-score: {z_score:.2f})"
            )
    
    # Register handler
    event_bus.on('BrokerMetric', handle_slippage_metric)
    
    return analyzer
