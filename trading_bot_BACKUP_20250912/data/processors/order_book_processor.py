#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OrderBookProcessor - Process order book data for market microstructure analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
import time
from collections import defaultdict

from trading_bot.data.processors.base_processor import DataProcessor

logger = logging.getLogger("OrderBookProcessor")

class OrderBookData:
    """Class to represent order book data at a specific timestamp."""
    
    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        last_trade_price: Optional[float] = None,
        last_trade_size: Optional[float] = None,
        source: Optional[str] = None
    ):
        """
        Initialize OrderBookData.
        
        Args:
            timestamp: Timestamp of the order book snapshot
            symbol: Trading symbol
            bids: List of (price, size) tuples for bids, sorted by price descending
            asks: List of (price, size) tuples for asks, sorted by price ascending
            last_trade_price: Last trade price (optional)
            last_trade_size: Last trade size (optional)
            source: Data source identifier (optional)
        """
        self.timestamp = timestamp
        self.symbol = symbol
        self.bids = sorted(bids, key=lambda x: x[0], reverse=True)  # Sort by price descending
        self.asks = sorted(asks, key=lambda x: x[0])  # Sort by price ascending
        self.last_trade_price = last_trade_price
        self.last_trade_size = last_trade_size
        self.source = source
        
        # Calculate mid price
        if bids and asks:
            self.mid_price = (bids[0][0] + asks[0][0]) / 2
        elif bids:
            self.mid_price = bids[0][0]
        elif asks:
            self.mid_price = asks[0][0]
        else:
            self.mid_price = None
        
        # Calculate spread
        if bids and asks:
            self.spread = asks[0][0] - bids[0][0]
            self.spread_pct = (self.spread / self.mid_price) * 100 if self.mid_price else None
        else:
            self.spread = None
            self.spread_pct = None
    
    def get_price_levels(self, side: str, levels: int = 10) -> List[Tuple[float, float]]:
        """
        Get price levels for a specific side.
        
        Args:
            side: 'bids' or 'asks'
            levels: Number of levels to return
            
        Returns:
            List of (price, size) tuples limited to specified levels
        """
        if side.lower() == 'bids':
            return self.bids[:levels]
        elif side.lower() == 'asks':
            return self.asks[:levels]
        else:
            raise ValueError("Side must be 'bids' or 'asks'")
    
    def get_cumulative_volume(self, side: str, price_delta: Optional[float] = None) -> float:
        """
        Get cumulative volume for a side up to a certain price delta from best price.
        
        Args:
            side: 'bids' or 'asks'
            price_delta: Price delta from best price (if None, use all levels)
            
        Returns:
            Cumulative volume
        """
        if side.lower() == 'bids':
            if price_delta is None:
                return sum(size for _, size in self.bids)
            else:
                threshold = self.bids[0][0] - price_delta if self.bids else 0
                return sum(size for price, size in self.bids if price >= threshold)
        elif side.lower() == 'asks':
            if price_delta is None:
                return sum(size for _, size in self.asks)
            else:
                threshold = self.asks[0][0] + price_delta if self.asks else float('inf')
                return sum(size for price, size in self.asks if price <= threshold)
        else:
            raise ValueError("Side must be 'bids' or 'asks'")
    
    def calculate_imbalance(self, levels: int = 5) -> float:
        """
        Calculate order book imbalance.
        
        Args:
            levels: Number of price levels to consider
            
        Returns:
            Order book imbalance (-1 to 1, where positive means more bids than asks)
        """
        bid_volume = sum(size for _, size in self.get_price_levels('bids', levels))
        ask_volume = sum(size for _, size in self.get_price_levels('asks', levels))
        
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            return (bid_volume - ask_volume) / total_volume
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert OrderBookData to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'bids': self.bids,
            'asks': self.asks,
            'mid_price': self.mid_price,
            'spread': self.spread,
            'spread_pct': self.spread_pct,
            'last_trade_price': self.last_trade_price,
            'last_trade_size': self.last_trade_size,
            'source': self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderBookData':
        """Create OrderBookData from dictionary."""
        return cls(
            timestamp=data['timestamp'],
            symbol=data['symbol'],
            bids=data['bids'],
            asks=data['asks'],
            last_trade_price=data.get('last_trade_price'),
            last_trade_size=data.get('last_trade_size'),
            source=data.get('source')
        )


class OrderBookProcessor(DataProcessor):
    """
    Processor for order book data to extract market microstructure features.
    """
    
    def __init__(self, name: str = "OrderBookProcessor", config: Optional[Dict[str, Any]] = None):
        """
        Initialize OrderBookProcessor.
        
        Args:
            name: Processor name
            config: Configuration dictionary
        """
        super().__init__(name, config)
        
        # Order book snapshots keyed by symbol
        self.order_book_snapshots = defaultdict(list)
        
        # Order book metrics keyed by symbol
        self.order_book_metrics = defaultdict(list)
        
        # Maximum number of snapshots to keep in memory
        self.max_snapshots = self.config.get('max_snapshots', 100)
        
        # Levels to analyze
        self.analysis_levels = self.config.get('analysis_levels', [1, 5, 10, 20])
        
        # Price grouping for volume profile
        self.price_grouping = self.config.get('price_grouping', 0.01)
        
        # Order flow analytics window
        self.order_flow_window = self.config.get('order_flow_window', 20)
        
        # Liquidity quality window
        self.liquidity_window = self.config.get('liquidity_window', 30)
    
    def init_from_config(self) -> None:
        """Initialize additional attributes from configuration."""
        # Define price levels to analyze (how many levels deep in the order book)
        self.price_levels = self.config.get('price_levels', 20)
        
        # Threshold for identifying large orders (as multiple of average)
        self.large_order_threshold = self.config.get('large_order_threshold', 5.0)
        
        # Volume window for calculating average volume
        self.volume_window = self.config.get('volume_window', 100)
        
        # Imbalance significance threshold
        self.imbalance_threshold = self.config.get('imbalance_threshold', 0.2)
        
        # Enable/disable specific analytics
        self.analyze_liquidity_cost = self.config.get('analyze_liquidity_cost', True)
        self.analyze_order_flow = self.config.get('analyze_order_flow', True)
        self.analyze_price_impact = self.config.get('analyze_price_impact', True)
        self.detect_spoofing = self.config.get('detect_spoofing', True)
    
    def process(self, data: Union[List[OrderBookData], pd.DataFrame]) -> pd.DataFrame:
        """
        Process order book data.
        
        Args:
            data: List of OrderBookData objects or DataFrame with order book snapshots
            
        Returns:
            DataFrame with processed order book features
        """
        # Convert data to order book snapshots if it's a DataFrame
        if isinstance(data, pd.DataFrame):
            snapshots = self._dataframe_to_snapshots(data)
        else:
            snapshots = data
        
        # Process each snapshot by symbol
        results = {}
        for snapshot in snapshots:
            symbol = snapshot.symbol
            
            # Store snapshot
            self.order_book_snapshots[symbol].append(snapshot)
            
            # Limit number of snapshots kept in memory
            if len(self.order_book_snapshots[symbol]) > self.max_snapshots:
                self.order_book_snapshots[symbol] = self.order_book_snapshots[symbol][-self.max_snapshots:]
            
            # Calculate order book metrics
            metrics = self._calculate_snapshot_metrics(snapshot)
            self.order_book_metrics[symbol].append(metrics)
            
            # Limit number of metrics kept in memory
            if len(self.order_book_metrics[symbol]) > self.max_snapshots:
                self.order_book_metrics[symbol] = self.order_book_metrics[symbol][-self.max_snapshots:]
        
        # Combine metrics into DataFrames by symbol
        for symbol, metrics_list in self.order_book_metrics.items():
            if symbol not in results:
                results[symbol] = pd.DataFrame(metrics_list)
        
        # If only one symbol, return that DataFrame
        if len(results) == 1:
            return list(results.values())[0]
        
        # Otherwise, return a dictionary of DataFrames
        return results
    
    def _dataframe_to_snapshots(self, df: pd.DataFrame) -> List[OrderBookData]:
        """
        Convert a DataFrame to OrderBookData snapshots.
        
        Args:
            df: DataFrame with order book data
            
        Returns:
            List of OrderBookData objects
        """
        snapshots = []
        
        # Check if the DataFrame has the required columns
        required_columns = ['timestamp', 'symbol', 'side', 'price', 'size']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return snapshots
        
        # Group by timestamp and symbol
        groups = df.groupby(['timestamp', 'symbol'])
        
        for (timestamp, symbol), group in groups:
            # Split into bids and asks
            bids_df = group[group['side'] == 'bid']
            asks_df = group[group['side'] == 'ask']
            
            # Convert to (price, size) tuples
            bids = [(row['price'], row['size']) for _, row in bids_df.iterrows()]
            asks = [(row['price'], row['size']) for _, row in asks_df.iterrows()]
            
            # Get last trade info if available
            last_trade_price = group['last_price'].iloc[0] if 'last_price' in group.columns else None
            last_trade_size = group['last_size'].iloc[0] if 'last_size' in group.columns else None
            
            # Create OrderBookData object
            snapshot = OrderBookData(
                timestamp=timestamp,
                symbol=symbol,
                bids=bids,
                asks=asks,
                last_trade_price=last_trade_price,
                last_trade_size=last_trade_size
            )
            
            snapshots.append(snapshot)
        
        return snapshots
    
    def _calculate_snapshot_metrics(self, snapshot: OrderBookData) -> Dict[str, Any]:
        """
        Calculate metrics for a single order book snapshot.
        
        Args:
            snapshot: OrderBookData object
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'timestamp': snapshot.timestamp,
            'symbol': snapshot.symbol,
            'mid_price': snapshot.mid_price,
            'spread': snapshot.spread,
            'spread_pct': snapshot.spread_pct,
            'best_bid': snapshot.bids[0][0] if snapshot.bids else None,
            'best_ask': snapshot.asks[0][0] if snapshot.asks else None,
            'best_bid_size': snapshot.bids[0][1] if snapshot.bids else 0,
            'best_ask_size': snapshot.asks[0][1] if snapshot.asks else 0
        }
        
        # Calculate imbalance at different levels
        for level in self.analysis_levels:
            metrics[f'imbalance_{level}'] = snapshot.calculate_imbalance(level)
        
        # Calculate cumulative volume at different price deltas
        for delta in [0.1, 0.5, 1.0, 5.0]:
            metrics[f'bid_volume_{delta}'] = snapshot.get_cumulative_volume('bids', delta)
            metrics[f'ask_volume_{delta}'] = snapshot.get_cumulative_volume('asks', delta)
        
        # Calculate total volume visible in order book
        metrics['total_bid_volume'] = snapshot.get_cumulative_volume('bids')
        metrics['total_ask_volume'] = snapshot.get_cumulative_volume('asks')
        
        # Calculate liquidity cost (how much price slippage for a given order size)
        if self.analyze_liquidity_cost:
            for test_size in [1.0, 5.0, 10.0, 50.0]:
                metrics[f'buy_price_impact_{test_size}'] = self._calculate_price_impact(snapshot, 'buy', test_size)
                metrics[f'sell_price_impact_{test_size}'] = self._calculate_price_impact(snapshot, 'sell', test_size)
        
        # Calculate order book slope (how quickly liquidity diminishes)
        metrics['bid_slope'] = self._calculate_slope(snapshot.bids)
        metrics['ask_slope'] = self._calculate_slope(snapshot.asks)
        
        return metrics
    
    def _calculate_price_impact(self, snapshot: OrderBookData, side: str, size: float) -> float:
        """
        Calculate price impact for a given order size.
        
        Args:
            snapshot: OrderBookData object
            side: 'buy' or 'sell'
            size: Order size
            
        Returns:
            Price impact as percentage of mid price
        """
        if snapshot.mid_price is None:
            return 0.0
        
        if side.lower() == 'buy':
            # Simulate buying into the asks
            remaining_size = size
            total_cost = 0.0
            
            for price, available_size in snapshot.asks:
                if remaining_size <= 0:
                    break
                
                executed_size = min(available_size, remaining_size)
                total_cost += executed_size * price
                remaining_size -= executed_size
            
            # If there's not enough liquidity, estimate with last known price
            if remaining_size > 0 and snapshot.asks:
                total_cost += remaining_size * snapshot.asks[-1][0]
            
            # Calculate average execution price
            avg_price = total_cost / size if size > 0 else snapshot.mid_price
            
            # Calculate impact as percentage
            return (avg_price / snapshot.mid_price - 1) * 100
            
        elif side.lower() == 'sell':
            # Simulate selling into the bids
            remaining_size = size
            total_proceeds = 0.0
            
            for price, available_size in snapshot.bids:
                if remaining_size <= 0:
                    break
                
                executed_size = min(available_size, remaining_size)
                total_proceeds += executed_size * price
                remaining_size -= executed_size
            
            # If there's not enough liquidity, estimate with last known price
            if remaining_size > 0 and snapshot.bids:
                total_proceeds += remaining_size * snapshot.bids[-1][0]
            
            # Calculate average execution price
            avg_price = total_proceeds / size if size > 0 else snapshot.mid_price
            
            # Calculate impact as percentage (negative for selling)
            return (avg_price / snapshot.mid_price - 1) * 100
            
        else:
            raise ValueError("Side must be 'buy' or 'sell'")
    
    def _calculate_slope(self, price_levels: List[Tuple[float, float]], depth: int = 5) -> float:
        """
        Calculate the slope of price levels.
        
        Args:
            price_levels: List of (price, size) tuples
            depth: Number of levels to consider
            
        Returns:
            Slope value (higher means more steep drop-off in liquidity)
        """
        if not price_levels or len(price_levels) < 2:
            return 0.0
        
        # Use min of available levels and requested depth
        depth = min(depth, len(price_levels))
        
        # Extract prices and sizes
        prices = [price for price, _ in price_levels[:depth]]
        sizes = [size for _, size in price_levels[:depth]]
        
        if len(prices) < 2:
            return 0.0
        
        # Calculate slope using linear regression
        try:
            # Normalize prices for stability
            mean_price = sum(prices) / len(prices)
            normalized_prices = [(p - mean_price) for p in prices]
            
            # Calculate coefficients
            x = np.array(normalized_prices)
            y = np.array(sizes)
            
            # Simple linear regression for slope
            slope = np.polyfit(x, y, 1)[0]
            return slope
        except:
            logger.warning("Failed to calculate slope")
            return 0.0
    
    def calculate_vwap(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate Volume Weighted Average Price from trades.
        
        Args:
            trades: List of trade dictionaries with 'price' and 'size' keys
            
        Returns:
            VWAP value
        """
        if not trades:
            return 0.0
            
        total_volume = sum(trade['size'] for trade in trades)
        if total_volume == 0:
            return 0.0
            
        vwap = sum(trade['price'] * trade['size'] for trade in trades) / total_volume
        return vwap
    
    def detect_large_orders(self, 
                           snapshots: List[OrderBookData], 
                           threshold_multiple: float = None) -> List[Dict[str, Any]]:
        """
        Detect large orders in order book snapshots.
        
        Args:
            snapshots: List of OrderBookData objects
            threshold_multiple: Multiple of average size to consider large (if None, use config)
            
        Returns:
            List of detected large orders
        """
        if not snapshots:
            return []
        
        # Use config threshold if not specified
        if threshold_multiple is None:
            threshold_multiple = self.large_order_threshold
        
        # Calculate average sizes
        all_bid_sizes = []
        all_ask_sizes = []
        
        for snapshot in snapshots:
            for _, size in snapshot.bids:
                all_bid_sizes.append(size)
            for _, size in snapshot.asks:
                all_ask_sizes.append(size)
        
        # Calculate average and standard deviation
        avg_bid_size = np.mean(all_bid_sizes) if all_bid_sizes else 0
        avg_ask_size = np.mean(all_ask_sizes) if all_ask_sizes else 0
        
        std_bid_size = np.std(all_bid_sizes) if len(all_bid_sizes) > 1 else 0
        std_ask_size = np.std(all_ask_sizes) if len(all_ask_sizes) > 1 else 0
        
        # Calculate threshold
        bid_threshold = avg_bid_size + (threshold_multiple * std_bid_size)
        ask_threshold = avg_ask_size + (threshold_multiple * std_ask_size)
        
        # Detect large orders
        large_orders = []
        
        for snapshot in snapshots:
            for price, size in snapshot.bids:
                if size > bid_threshold:
                    large_orders.append({
                        'timestamp': snapshot.timestamp,
                        'symbol': snapshot.symbol,
                        'side': 'bid',
                        'price': price,
                        'size': size,
                        'threshold': bid_threshold
                    })
            
            for price, size in snapshot.asks:
                if size > ask_threshold:
                    large_orders.append({
                        'timestamp': snapshot.timestamp,
                        'symbol': snapshot.symbol,
                        'side': 'ask',
                        'price': price,
                        'size': size,
                        'threshold': ask_threshold
                    })
        
        return large_orders
    
    def calculate_depth_imbalance_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate depth imbalance technical indicators from order book metrics DataFrame.
        
        Args:
            df: DataFrame with order book metrics
            
        Returns:
            DataFrame with added depth imbalance indicators
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate depth imbalance at different levels
        for level in self.analysis_levels:
            col_name = f'imbalance_{level}'
            if col_name in result.columns:
                # Depth imbalance oscillator (DIO)
                result[f'dio_{level}'] = result[col_name].rolling(window=10).mean()
                
                # DIO signal line
                result[f'dio_{level}_signal'] = result[f'dio_{level}'].rolling(window=5).mean()
                
                # DIO crossover signal
                result[f'dio_{level}_cross'] = np.where(
                    (result[f'dio_{level}'] > result[f'dio_{level}_signal']) & 
                    (result[f'dio_{level}'].shift(1) <= result[f'dio_{level}_signal'].shift(1)),
                    1,  # Bullish crossover
                    np.where(
                        (result[f'dio_{level}'] < result[f'dio_{level}_signal']) & 
                        (result[f'dio_{level}'].shift(1) >= result[f'dio_{level}_signal'].shift(1)),
                        -1,  # Bearish crossover
                        0    # No crossover
                    )
                )
                
                # Calculate divergence with price
                result[f'dio_{level}_divergence'] = self._calculate_indicator_divergence(
                    result['mid_price'], 
                    result[f'dio_{level}']
                )
        
        # Calculate buy/sell pressure indicator
        if 'total_bid_volume' in result.columns and 'total_ask_volume' in result.columns:
            # Buy/Sell Pressure
            result['buy_sell_pressure'] = (result['total_bid_volume'] - result['total_ask_volume']) / \
                                         (result['total_bid_volume'] + result['total_ask_volume'])
            
            # Smoothed Buy/Sell Pressure
            result['buy_sell_pressure_smooth'] = result['buy_sell_pressure'].rolling(window=10).mean()
            
            # Buy/Sell Pressure momentum
            result['bsp_momentum'] = result['buy_sell_pressure_smooth'] - result['buy_sell_pressure_smooth'].shift(5)
        
        # Calculate liquidity depletion rate if price impact columns exist
        if 'buy_price_impact_10.0' in result.columns and 'buy_price_impact_10.0' in result.columns:
            # Liquidity depletion indicator
            result['liquidity_depletion'] = result['buy_price_impact_10.0'].rolling(window=10).mean() + \
                                          abs(result['sell_price_impact_10.0'].rolling(window=10).mean())
            
            # Normalize liquidity depletion
            mean_depletion = result['liquidity_depletion'].rolling(window=30).mean()
            std_depletion = result['liquidity_depletion'].rolling(window=30).std()
            
            result['liquidity_depletion_z'] = (result['liquidity_depletion'] - mean_depletion) / \
                                             std_depletion.replace(0, 1)  # Avoid div by zero
        
        # Calculate order flow indicators
        if 'bid_slope' in result.columns and 'ask_slope' in result.columns:
            # Slope ratio
            result['slope_ratio'] = result['bid_slope'] / result['ask_slope'].replace(0, 1e-10)
            
            # Normalize slope ratio
            result['slope_ratio_normalized'] = (result['slope_ratio'] - 
                                             result['slope_ratio'].rolling(window=20).mean()) / \
                                             result['slope_ratio'].rolling(window=20).std().replace(0, 1)
        
        return result
    
    def _calculate_indicator_divergence(self, price: pd.Series, indicator: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate divergence between price and an indicator.
        
        Args:
            price: Price series
            indicator: Indicator series
            window: Window for checking divergence
            
        Returns:
            Series with divergence values (0=no divergence, 1=bullish, -1=bearish)
        """
        # Initialize result
        divergence = pd.Series(0, index=price.index)
        
        # Need enough data for the window
        if len(price) < window:
            return divergence
        
        # Calculate rolling max/min for price and indicator
        price_highpoint = price.rolling(window=window).max()
        price_lowpoint = price.rolling(window=window).min()
        
        indicator_highpoint = indicator.rolling(window=window).max()
        indicator_lowpoint = indicator.rolling(window=window).min()
        
        # Detect bearish divergence (price high but indicator lower)
        bearish_divergence = (
            (price > 0.95 * price_highpoint) &  # Price near high
            (indicator < 0.7 * indicator_highpoint)  # Indicator not at high
        )
        
        # Detect bullish divergence (price low but indicator higher)
        bullish_divergence = (
            (price < 1.05 * price_lowpoint) &  # Price near low
            (indicator > 1.3 * indicator_lowpoint)  # Indicator not at low
        )
        
        # Set divergence values
        divergence[bearish_divergence] = -1
        divergence[bullish_divergence] = 1
        
        return divergence
    
    def get_latest_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest order book metrics for a specific symbol.
        
        Args:
            symbol: Symbol to get metrics for
            
        Returns:
            Dictionary of latest metrics or None if not available
        """
        if symbol in self.order_book_metrics and self.order_book_metrics[symbol]:
            return self.order_book_metrics[symbol][-1]
        return None
    
    def get_current_market_microstructure(self, symbol: str) -> Dict[str, Any]:
        """
        Get current market microstructure analysis for a specific symbol.
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Dictionary with market microstructure analysis
        """
        # Get latest metrics and snapshots
        latest_metrics = self.get_latest_metrics(symbol)
        if not latest_metrics:
            return {'status': 'No data available'}
        
        snapshots = self.order_book_snapshots.get(symbol, [])
        if not snapshots:
            return {'status': 'No order book snapshots available'}
        
        # Create basic analysis
        analysis = {
            'timestamp': latest_metrics['timestamp'],
            'symbol': symbol,
            'mid_price': latest_metrics['mid_price'],
            'spread_bps': latest_metrics['spread_pct'] * 100 if latest_metrics.get('spread_pct') else None,
        }
        
        # Add order book imbalance
        for level in self.analysis_levels:
            if f'imbalance_{level}' in latest_metrics:
                analysis[f'imbalance_{level}_levels'] = latest_metrics[f'imbalance_{level}']
        
        # Determine overall imbalance strength and direction
        imbalance_cols = [col for col in latest_metrics.keys() if col.startswith('imbalance_')]
        if imbalance_cols:
            # Take average of imbalances
            avg_imbalance = np.mean([latest_metrics[col] for col in imbalance_cols])
            
            # Determine strength and direction
            if abs(avg_imbalance) < 0.05:
                imbalance_state = "Balanced"
            elif avg_imbalance > 0.2:
                imbalance_state = "Strong Buy Pressure"
            elif avg_imbalance > 0.05:
                imbalance_state = "Mild Buy Pressure"
            elif avg_imbalance < -0.2:
                imbalance_state = "Strong Sell Pressure"
            else:
                imbalance_state = "Mild Sell Pressure"
                
            analysis['order_book_state'] = imbalance_state
            analysis['imbalance_score'] = avg_imbalance
        
        # Add liquidity analysis
        if 'buy_price_impact_10.0' in latest_metrics and 'sell_price_impact_10.0' in latest_metrics:
            analysis['buy_price_impact_bps'] = latest_metrics['buy_price_impact_10.0'] * 100
            analysis['sell_price_impact_bps'] = latest_metrics['sell_price_impact_10.0'] * 100
            
            # Calculate average impact
            avg_impact = (analysis['buy_price_impact_bps'] + abs(analysis['sell_price_impact_bps'])) / 2
            
            # Determine liquidity state
            if avg_impact < 5:
                liquidity_state = "Highly Liquid"
            elif avg_impact < 15:
                liquidity_state = "Normally Liquid"
            elif avg_impact < 50:
                liquidity_state = "Moderately Illiquid"
            else:
                liquidity_state = "Highly Illiquid"
                
            analysis['liquidity_state'] = liquidity_state
            analysis['avg_price_impact_bps'] = avg_impact
        
        # Detect large orders
        large_orders = self.detect_large_orders(snapshots[-10:])
        if large_orders:
            analysis['large_orders'] = len(large_orders)
            analysis['large_orders_details'] = large_orders
        else:
            analysis['large_orders'] = 0
        
        # Add market recommendation
        if 'order_book_state' in analysis and 'liquidity_state' in analysis:
            if analysis['order_book_state'] in ["Strong Buy Pressure", "Mild Buy Pressure"] and \
               analysis['liquidity_state'] in ["Highly Liquid", "Normally Liquid"]:
                recommendation = "Consider Long Position"
            elif analysis['order_book_state'] in ["Strong Sell Pressure", "Mild Sell Pressure"] and \
                 analysis['liquidity_state'] in ["Highly Liquid", "Normally Liquid"]:
                recommendation = "Consider Short Position"
            elif analysis['liquidity_state'] in ["Moderately Illiquid", "Highly Illiquid"]:
                recommendation = "Caution - Low Liquidity"
            else:
                recommendation = "Neutral - No Clear Signal"
                
            analysis['recommendation'] = recommendation
        
        return analysis 