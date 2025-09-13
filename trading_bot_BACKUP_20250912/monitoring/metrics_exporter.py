#!/usr/bin/env python3
"""
Trading Bot Metrics Exporter

This module exports trading bot metrics to Prometheus.
It collects metrics from the trading system and exposes them via HTTP endpoint.
"""

import time
import logging
from typing import Dict, Any, List, Optional
import threading
import argparse
from datetime import datetime, timedelta

from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
from prometheus_client import REGISTRY, PROCESS_COLLECTOR, PLATFORM_COLLECTOR

# Import your trading bot modules
from trading_bot.risk_manager import RiskManager
from trading_bot.options_risk_manager import OptionsRiskManager
from trading_bot.adapters.multi_asset_adapter import MultiAssetAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("metrics_exporter")

class TradingBotMetricsExporter:
    """
    Exports trading bot metrics to Prometheus.
    """
    
    def __init__(
        self, 
        multi_asset_adapter: MultiAssetAdapter,
        risk_manager: RiskManager,
        options_risk_manager: Optional[OptionsRiskManager] = None,
        collection_interval: int = 15
    ):
        """
        Initialize the metrics exporter.
        
        Args:
            multi_asset_adapter: Instance of MultiAssetAdapter
            risk_manager: Instance of RiskManager
            options_risk_manager: Optional instance of OptionsRiskManager
            collection_interval: Metrics collection interval in seconds
        """
        self.multi_asset_adapter = multi_asset_adapter
        self.risk_manager = risk_manager
        self.options_risk_manager = options_risk_manager
        self.collection_interval = collection_interval
        
        # Flag to control the background thread
        self.running = False
        self.collection_thread = None
        
        # Initialize metrics
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize all Prometheus metrics."""
        # General account metrics
        self.account_balance = Gauge(
            'trading_bot_account_balance', 
            'Current account balance'
        )
        self.account_equity = Gauge(
            'trading_bot_account_equity', 
            'Current account equity'
        )
        self.margin_used = Gauge(
            'trading_bot_margin_used', 
            'Current margin used'
        )
        self.margin_available = Gauge(
            'trading_bot_margin_available', 
            'Current margin available'
        )
        
        # Trading metrics
        self.position_count = Gauge(
            'trading_bot_position_count', 
            'Number of open positions',
            ['asset_class']
        )
        self.position_value = Gauge(
            'trading_bot_position_value', 
            'Total value of open positions',
            ['asset_class']
        )
        self.position_risk = Gauge(
            'trading_bot_position_risk', 
            'Current position risk'
        )
        self.risk_threshold = Gauge(
            'trading_bot_risk_threshold', 
            'Current risk threshold'
        )
        
        # P&L metrics
        self.daily_pnl = Gauge(
            'trading_bot_daily_pnl', 
            'Profit and loss for the current day'
        )
        self.cumulative_pnl = Gauge(
            'trading_bot_cumulative_pnl', 
            'Cumulative profit and loss'
        )
        
        # Trade execution metrics
        self.order_execution_count = Counter(
            'trading_bot_order_execution_total', 
            'Total number of order executions',
            ['asset_class', 'side', 'result']
        )
        self.order_execution_time = Histogram(
            'trading_bot_order_execution_time_seconds', 
            'Time to execute an order',
            ['asset_class', 'side'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
        )
        
        # Data collection metrics
        self.last_data_collection = Gauge(
            'trading_bot_last_data_collection_timestamp', 
            'Timestamp of the last successful data collection'
        )
        self.data_collection_errors = Counter(
            'trading_bot_data_collection_errors_total', 
            'Total number of data collection errors',
            ['source']
        )
        
        # API metrics
        self.api_requests = Counter(
            'trading_bot_api_requests_total', 
            'Total number of API requests',
            ['endpoint', 'method']
        )
        self.api_errors = Counter(
            'trading_bot_api_errors_total', 
            'Total number of API errors',
            ['endpoint', 'method', 'status']
        )
        
        # Options-specific metrics
        if self.options_risk_manager:
            self.options_delta_exposure = Gauge(
                'trading_bot_options_delta_exposure', 
                'Current options delta exposure'
            )
            self.options_gamma_exposure = Gauge(
                'trading_bot_options_gamma_exposure', 
                'Current options gamma exposure'
            )
            self.options_theta_exposure = Gauge(
                'trading_bot_options_theta_exposure', 
                'Current options theta exposure'
            )
            self.options_vega_exposure = Gauge(
                'trading_bot_options_vega_exposure', 
                'Current options vega exposure'
            )
            self.options_position_count = Gauge(
                'trading_bot_options_position_count', 
                'Number of open options positions',
                ['option_type', 'strategy']
            )
        
        # System info
        self.system_info = Info(
            'trading_bot_system_info', 
            'Trading bot system information'
        )
        self.system_info.info({
            'version': '1.0.0',
            'name': 'trading_bot',
            'start_time': datetime.now().isoformat()
        })
    
    def collect_metrics(self):
        """Collect all metrics from the trading bot."""
        try:
            logger.info("Collecting metrics...")
            
            # Collect account metrics
            account_info = self.multi_asset_adapter.get_account_info()
            self.account_balance.set(account_info.get('balance', 0))
            self.account_equity.set(account_info.get('equity', 0))
            self.margin_used.set(account_info.get('margin_used', 0))
            self.margin_available.set(account_info.get('margin_available', 0))
            
            # Collect position metrics
            positions = self.multi_asset_adapter.get_all_positions()
            position_count_by_asset = {}
            position_value_by_asset = {}
            
            for position in positions:
                asset_class = position.get('asset_class', 'unknown')
                position_count_by_asset[asset_class] = position_count_by_asset.get(asset_class, 0) + 1
                position_value_by_asset[asset_class] = position_value_by_asset.get(asset_class, 0) + position.get('current_value', 0)
            
            for asset_class, count in position_count_by_asset.items():
                self.position_count.labels(asset_class=asset_class).set(count)
                
            for asset_class, value in position_value_by_asset.items():
                self.position_value.labels(asset_class=asset_class).set(value)
            
            # Collect risk metrics
            risk_data = self.risk_manager.get_risk_metrics()
            self.position_risk.set(risk_data.get('position_risk', 0))
            self.risk_threshold.set(risk_data.get('risk_threshold', 0))
            
            # Collect P&L metrics
            performance = self.multi_asset_adapter.get_performance_metrics()
            self.daily_pnl.set(performance.get('daily_pnl', 0))
            self.cumulative_pnl.set(performance.get('cumulative_pnl', 0))
            
            # Update data collection timestamp
            self.last_data_collection.set(time.time())
            
            # Collect options-specific metrics if available
            if self.options_risk_manager:
                self._collect_options_metrics()
                
            logger.info("Metrics collection completed")
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}", exc_info=True)
    
    def _collect_options_metrics(self):
        """Collect options-specific metrics."""
        try:
            # Get portfolio Greeks
            portfolio_greeks = self.options_risk_manager.get_portfolio_greeks()
            
            # Set metrics
            self.options_delta_exposure.set(portfolio_greeks.get('delta', 0))
            self.options_gamma_exposure.set(portfolio_greeks.get('gamma', 0))
            self.options_theta_exposure.set(portfolio_greeks.get('theta', 0))
            self.options_vega_exposure.set(portfolio_greeks.get('vega', 0))
            
            # Count options positions by type and strategy
            options_positions = [
                p for p in self.multi_asset_adapter.get_all_positions() 
                if p.get('asset_class') == 'OPTIONS'
            ]
            
            # Reset counters
            for option_type in ['CALL', 'PUT']:
                for strategy in ['SINGLE_LEG', 'VERTICAL_SPREAD', 'IRON_CONDOR', 'BUTTERFLY', 
                               'CALENDAR_SPREAD', 'STRADDLE', 'STRANGLE']:
                    self.options_position_count.labels(
                        option_type=option_type, 
                        strategy=strategy
                    ).set(0)
            
            # Count positions
            for position in options_positions:
                option_type = position.get('metadata', {}).get('option_type', 'UNKNOWN')
                strategy = position.get('metadata', {}).get('strategy', 'SINGLE_LEG')
                
                self.options_position_count.labels(
                    option_type=option_type, 
                    strategy=strategy
                ).inc()
                
            logger.info("Options metrics collection completed")
            
        except Exception as e:
            logger.error(f"Error collecting options metrics: {str(e)}", exc_info=True)
    
    def start_metrics_server(self, port: int = 8000):
        """
        Start the Prometheus metrics server.
        
        Args:
            port: HTTP port to expose metrics on
        """
        try:
            start_http_server(port)
            logger.info(f"Metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {str(e)}", exc_info=True)
            raise
    
    def start_collecting(self):
        """Start collecting metrics in a background thread."""
        if self.running:
            logger.warning("Metrics collection already running")
            return
        
        self.running = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop, 
            daemon=True,
            name="metrics-collector"
        )
        self.collection_thread.start()
        logger.info("Started metrics collection thread")
    
    def stop_collecting(self):
        """Stop the metrics collection thread."""
        if not self.running:
            logger.warning("Metrics collection not running")
            return
        
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=60)
            logger.info("Stopped metrics collection thread")
    
    def _collection_loop(self):
        """Background thread for periodic metrics collection."""
        while self.running:
            try:
                self.collect_metrics()
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {str(e)}", exc_info=True)
            
            # Sleep until next collection
            time.sleep(self.collection_interval)


def main():
    """Main function to run the metrics exporter as a standalone process."""
    parser = argparse.ArgumentParser(description="Trading Bot Metrics Exporter")
    parser.add_argument("--port", type=int, default=8000, help="Metrics server port")
    parser.add_argument("--interval", type=int, default=15, help="Metrics collection interval in seconds")
    args = parser.parse_args()
    
    try:
        # Initialize your trading bot components
        from trading_bot.config import load_config
        from trading_bot.adapters.multi_asset_adapter import MultiAssetAdapter
        from trading_bot.risk_manager import RiskManager
        from trading_bot.options_risk_manager import OptionsRiskManager
        
        config = load_config()
        
        # Initialize components
        multi_asset_adapter = MultiAssetAdapter(config)
        risk_manager = RiskManager(config['risk'])
        options_risk_manager = OptionsRiskManager(multi_asset_adapter, risk_manager, config['options'])
        
        # Initialize metrics exporter
        exporter = TradingBotMetricsExporter(
            multi_asset_adapter=multi_asset_adapter,
            risk_manager=risk_manager,
            options_risk_manager=options_risk_manager,
            collection_interval=args.interval
        )
        
        # Start the server
        exporter.start_metrics_server(args.port)
        
        # Start collecting metrics
        exporter.start_collecting()
        
        # Keep the main thread alive
        logger.info("Metrics exporter running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Stopping metrics exporter...")
    except Exception as e:
        logger.error(f"Error in metrics exporter: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 