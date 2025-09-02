#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HFT Strategy Utilities

This module contains utility functions for the High-Frequency Trading (HFT) strategy,
focusing on orderbook analysis, statistical calculations, and execution metrics.
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

def analyze_orderbook(orderbook: Dict[str, Any], depth: int = 10) -> Dict[str, Any]:
    """
    Analyze orderbook data to extract HFT-relevant metrics.
    
    Args:
        orderbook: Dictionary containing bids and asks
        depth: Number of price levels to analyze
        
    Returns:
        Dictionary of orderbook metrics
    """
    results = {
        "bid_ask_spread": 0.0,
        "bid_ask_spread_pct": 0.0,
        "book_imbalance": 0.0,
        "liquidity_score": 0.0,
        "buy_pressure": 0.0,
        "sell_pressure": 0.0,
        "best_bid": 0.0,
        "best_ask": 0.0,
        "mid_price": 0.0,
        "weighted_mid_price": 0.0,
        "total_bid_volume": 0.0,
        "total_ask_volume": 0.0,
    }
    
    # Check for valid orderbook
    if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
        return results
    
    bids = orderbook['bids']
    asks = orderbook['asks']
    
    if not bids or not asks:
        return results
    
    # Limit to specified depth
    bids = bids[:depth] if len(bids) > depth else bids
    asks = asks[:depth] if len(asks) > depth else asks
    
    # Extract best bid and ask
    best_bid = bids[0][0] if bids else 0
    best_ask = asks[0][0] if asks else 0
    
    if best_bid <= 0 or best_ask <= 0:
        return results
    
    # Calculate basic metrics
    results["best_bid"] = best_bid
    results["best_ask"] = best_ask
    results["bid_ask_spread"] = best_ask - best_bid
    results["bid_ask_spread_pct"] = (best_ask - best_bid) / best_bid * 100.0
    results["mid_price"] = (best_bid + best_ask) / 2
    
    # Calculate volume-weighted metrics
    total_bid_volume = sum(level[1] for level in bids)
    total_ask_volume = sum(level[1] for level in asks)
    
    results["total_bid_volume"] = total_bid_volume
    results["total_ask_volume"] = total_ask_volume
    
    # Book imbalance (positive means more bids than asks)
    if total_bid_volume + total_ask_volume > 0:
        results["book_imbalance"] = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
    
    # Weighted mid price
    weighted_bid = sum(level[0] * level[1] for level in bids) / total_bid_volume if total_bid_volume > 0 else best_bid
    weighted_ask = sum(level[0] * level[1] for level in asks) / total_ask_volume if total_ask_volume > 0 else best_ask
    results["weighted_mid_price"] = (weighted_bid + weighted_ask) / 2
    
    # Buying and selling pressure
    close_to_best_bid_volume = sum(level[1] for level in bids if level[0] >= best_bid * 0.99)
    close_to_best_ask_volume = sum(level[1] for level in asks if level[0] <= best_ask * 1.01)
    
    results["buy_pressure"] = close_to_best_bid_volume / total_bid_volume if total_bid_volume > 0 else 0
    results["sell_pressure"] = close_to_best_ask_volume / total_ask_volume if total_ask_volume > 0 else 0
    
    # Liquidity score (higher is better)
    results["liquidity_score"] = (total_bid_volume + total_ask_volume) / results["bid_ask_spread"] if results["bid_ask_spread"] > 0 else 0
    
    return results

def calculate_z_score(price_history: List[float], current_price: float) -> float:
    """
    Calculate the z-score for the current price relative to recent history.
    
    Args:
        price_history: List of historical prices
        current_price: Current price
        
    Returns:
        Z-score value
    """
    if not price_history or len(price_history) < 10:
        return 0.0
    
    # Calculate mean and standard deviation
    prices = np.array(price_history)
    mean = np.mean(prices)
    std_dev = np.std(prices)
    
    # Avoid division by zero
    if std_dev == 0:
        return 0.0
    
    # Calculate z-score
    z_score = (current_price - mean) / std_dev
    
    return z_score

def detect_price_spikes(price_history: List[float], threshold: float = 3.0) -> bool:
    """
    Detect unusual price spikes (potential flash crashes or pumps).
    
    Args:
        price_history: List of recent price points
        threshold: Z-score threshold for spike detection
        
    Returns:
        True if spike detected, False otherwise
    """
    if len(price_history) < 20:
        return False
    
    # Calculate returns
    prices = np.array(price_history)
    returns = np.diff(prices) / prices[:-1]
    
    # Calculate z-scores of returns
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return False
    
    latest_return = returns[-1]
    z_score = (latest_return - mean_return) / std_return
    
    return abs(z_score) > threshold

def calculate_tick_size(price: float) -> float:
    """
    Calculate the tick size (minimum price increment) based on price range.
    
    Args:
        price: Current price
        
    Returns:
        Estimated tick size
    """
    # These are common tick sizes for different price ranges in crypto markets
    if price < 0.1:
        return 0.00001
    elif price < 1:
        return 0.0001
    elif price < 10:
        return 0.001
    elif price < 100:
        return 0.01
    elif price < 1000:
        return 0.1
    else:
        return 1.0

def calculate_execution_efficiency(order_times: List[float]) -> Dict[str, float]:
    """
    Calculate execution efficiency metrics.
    
    Args:
        order_times: List of order execution times in milliseconds
        
    Returns:
        Dictionary of efficiency metrics
    """
    if not order_times or len(order_times) < 5:
        return {"avg_execution_time": 0, "median_execution_time": 0, "95th_percentile": 0}
    
    times = np.array(order_times)
    
    return {
        "avg_execution_time": np.mean(times),
        "median_execution_time": np.median(times),
        "95th_percentile": np.percentile(times, 95)
    }

def check_for_latency_issues(recent_latencies: List[float], threshold_ms: float) -> bool:
    """
    Check for potential latency issues that could affect HFT performance.
    
    Args:
        recent_latencies: List of recent execution latencies in milliseconds
        threshold_ms: Threshold for acceptable latency
        
    Returns:
        True if latency issues detected, False otherwise
    """
    if not recent_latencies or len(recent_latencies) < 5:
        return False
    
    # Calculate recent average latency
    avg_latency = sum(recent_latencies) / len(recent_latencies)
    
    # Check if latency exceeds threshold
    return avg_latency > threshold_ms

def detect_market_regime(returns: List[float], volumes: List[float]) -> str:
    """
    Detect the current market regime for adaptive HFT behavior.
    
    Args:
        returns: List of recent price returns
        volumes: List of recent volumes
        
    Returns:
        Market regime as string: "low_volatility", "high_volatility", "trending", "ranging"
    """
    if len(returns) < 20 or len(volumes) < 20:
        return "unknown"
    
    # Calculate volatility
    volatility = np.std(returns) * np.sqrt(len(returns))
    
    # Calculate average volume
    avg_volume = np.mean(volumes)
    recent_avg_volume = np.mean(volumes[-5:])
    
    # Calculate trend strength
    returns_arr = np.array(returns)
    positive_returns = sum(1 for r in returns_arr if r > 0)
    negative_returns = len(returns_arr) - positive_returns
    trend_strength = abs(positive_returns - negative_returns) / len(returns_arr)
    
    # Detect regime
    high_vol_threshold = 0.02  # 2% annualized volatility threshold
    
    if volatility > high_vol_threshold:
        if trend_strength > 0.6:
            return "trending"
        else:
            return "high_volatility"
    else:
        if recent_avg_volume > avg_volume * 1.2:
            return "building_momentum"
        else:
            return "low_volatility"
