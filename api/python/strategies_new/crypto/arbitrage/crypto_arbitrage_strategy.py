#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Arbitrage Strategy

This strategy is designed to capitalize on price differences between exchanges (cross-exchange)
or between different trading pairs (triangular). It continuously monitors price spreads
and executes trades when the price difference exceeds the transaction costs and provides
a risk-adjusted profit opportunity.

Key characteristics:
- Risk-free or low-risk trades
- Very short holding periods
- Requires multi-exchange connectivity
- Highly sensitive to transaction costs and latency
- Works best in markets with good liquidity
- Can be profitable in any market regime but requires proper execution
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trading_bot.strategies_new.crypto.base import CryptoBaseStrategy, CryptoSession
from trading_bot.core.events import Event, EventType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.factory.registry import register_strategy

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="CryptoArbitrageStrategy",
    market_type="crypto",
    description="Arbitrage strategy for crypto markets exploiting price differences between exchanges or trading pairs",
    timeframes=["M1", "M5"],  # Arbitrage requires quick execution, so shorter timeframes
    parameters={
        # Arbitrage parameters
        "arbitrage_type": {"type": "str", "default": "cross_exchange", "enum": ["cross_exchange", "triangular"]},
        "min_profit_threshold_pct": {"type": "float", "default": 0.005, "min": 0.001, "max": 0.05},
        
        # Cross-exchange specific
        "exchanges": {"type": "list", "default": ["Binance", "Coinbase", "Kraken"]},
        "max_price_age_seconds": {"type": "int", "default": 5, "min": 1, "max": 30},
        
        # Triangular specific
        "triangle_pairs": {"type": "list", "default": []},  # Will be populated dynamically
        "base_currency": {"type": "str", "default": "USDT"},
        
        # Execution parameters
        "execution_timeout_seconds": {"type": "int", "default": 3, "min": 1, "max": 10},
        "simultaneous_opportunities": {"type": "int", "default": 2, "min": 1, "max": 5},
        
        # Risk management
        "max_exposure_per_exchange_pct": {"type": "float", "default": 0.25, "min": 0.1, "max": 0.5},
        "position_size_pct": {"type": "float", "default": 0.05, "min": 0.01, "max": 0.2},
        "max_slippage_pct": {"type": "float", "default": 0.001, "min": 0.0001, "max": 0.01},
    }
)
class CryptoArbitrageStrategy(CryptoBaseStrategy):
    """
    An arbitrage strategy for cryptocurrency markets.
    
    This strategy:
    1. Monitors price differentials between exchanges (cross-exchange) or trading pairs (triangular)
    2. Calculates profit opportunities accounting for fees, slippage, and transfer costs
    3. Executes simultaneous trades when profitable opportunities are identified
    4. Manages risk by limiting exposure and position sizes
    5. Continuously optimizes based on execution performance and market conditions
    """
    
    def __init__(self, session: CryptoSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the crypto arbitrage strategy.
        
        Args:
            session: Crypto trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        super().__init__(session, data_pipeline, parameters)
        
        # Arbitrage-specific state
        self.arbitrage_type = self.parameters.get("arbitrage_type", "cross_exchange")
        self.min_profit_threshold = self.parameters.get("min_profit_threshold_pct", 0.005)
        self.exchanges = self.parameters.get("exchanges", ["Binance", "Coinbase", "Kraken"])
        self.max_price_age_seconds = self.parameters.get("max_price_age_seconds", 5)
        
        # Exchange price data
        self.exchange_prices = {exchange: {"price": 0.0, "timestamp": None} for exchange in self.exchanges}
        
        # For triangular arbitrage
        self.base_currency = self.parameters.get("base_currency", "USDT")
        self.triangle_pairs = self.parameters.get("triangle_pairs", [])
        if not self.triangle_pairs and self.arbitrage_type == "triangular":
            # Default triangle setup: Base -> BTC -> ETH -> Base
            self.triangle_pairs = [
                f"BTC-{self.base_currency}",
                "ETH-BTC",
                f"ETH-{self.base_currency}"
            ]
        
        # Opportunity tracking
        self.active_opportunities = []
        self.recent_opportunities = []
        self.opportunity_history = []
        
        # Performance tracking
        self.successful_arbitrages = 0
        self.failed_arbitrages = 0
        self.total_profit = 0.0
        self.total_fees = 0.0
        
        logger.info(f"Initialized {self.arbitrage_type} arbitrage strategy with min profit threshold: {self.min_profit_threshold:.4f}")
        if self.arbitrage_type == "cross_exchange":
            logger.info(f"Monitoring exchanges: {', '.join(self.exchanges)}")
        else:  # triangular
            logger.info(f"Monitoring triangle pairs: {', '.join(self.triangle_pairs)}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators for arbitrage opportunities.
        
        For cross-exchange arbitrage: price differentials between exchanges
        For triangular arbitrage: conversion rates across trading pairs
        
        Args:
            data: Market data DataFrame (not heavily used in arbitrage strategies)
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if self.arbitrage_type == "cross_exchange":
            indicators = self._calculate_cross_exchange_indicators()
        else:  # triangular
            indicators = self._calculate_triangular_indicators()
            
        return indicators
    
    def _calculate_cross_exchange_indicators(self) -> Dict[str, Any]:
        """Calculate indicators for cross-exchange arbitrage."""
        indicators = {
            "price_differentials": [],
            "best_opportunity": None,
            "max_spread_pct": 0.0,
            "current_spreads": {},
            "fees_adjusted_spreads": {},
        }
        
        now = datetime.utcnow()
        valid_prices = {}
        
        # Filter for valid (recent) prices
        for exchange, data in self.exchange_prices.items():
            if data["timestamp"] is not None:
                age_seconds = (now - data["timestamp"]).total_seconds()
                if age_seconds <= self.max_price_age_seconds:
                    valid_prices[exchange] = data["price"]
        
        # Need at least two exchanges with valid prices
        if len(valid_prices) < 2:
            return indicators
        
        # Calculate price differentials between all exchange pairs
        exchanges = list(valid_prices.keys())
        for i in range(len(exchanges)):
            for j in range(i+1, len(exchanges)):
                exchange1 = exchanges[i]
                exchange2 = exchanges[j]
                price1 = valid_prices[exchange1]
                price2 = valid_prices[exchange2]
                
                # Skip if prices are zero (invalid)
                if price1 <= 0 or price2 <= 0:
                    continue
                
                # Calculate raw spread
                spread_pct = abs(price1 - price2) / min(price1, price2)
                
                # Calculate fee-adjusted spread (assuming two trades: buy & sell)
                fee_rate = self.session.trading_fees  # Default fee
                fee_adjusted_spread = spread_pct - (2 * fee_rate)
                
                # Track spread information
                pair_key = f"{exchange1}-{exchange2}"
                indicators["current_spreads"][pair_key] = spread_pct
                indicators["fees_adjusted_spreads"][pair_key] = fee_adjusted_spread
                
                # Track the max spread
                if spread_pct > indicators["max_spread_pct"]:
                    indicators["max_spread_pct"] = spread_pct
                
                # Add to differentials
                indicators["price_differentials"].append({
                    "pair": pair_key,
                    "buy_exchange": exchange1 if price1 < price2 else exchange2,
                    "sell_exchange": exchange2 if price1 < price2 else exchange1,
                    "buy_price": min(price1, price2),
                    "sell_price": max(price1, price2),
                    "spread_pct": spread_pct,
                    "fee_adjusted_spread": fee_adjusted_spread,
                    "profitable": fee_adjusted_spread > self.min_profit_threshold
                })
        
        # Sort by fee-adjusted spread (descending)
        indicators["price_differentials"].sort(key=lambda x: x["fee_adjusted_spread"], reverse=True)
        
        # Identify best opportunity
        profitable_opportunities = [
            diff for diff in indicators["price_differentials"] 
            if diff["fee_adjusted_spread"] > self.min_profit_threshold
        ]
        
        if profitable_opportunities:
            indicators["best_opportunity"] = profitable_opportunities[0]
        
        return indicators
    
    def _calculate_triangular_indicators(self) -> Dict[str, Any]:
        """Calculate indicators for triangular arbitrage."""
        indicators = {
            "triangle_rates": [],
            "best_opportunity": None,
            "max_profit_pct": 0.0,
        }
        
        # For triangular arbitrage, we need at least 3 pairs to form a triangle
        if len(self.triangle_pairs) < 3:
            return indicators
        
        # Get latest prices for triangle pairs
        triangle_prices = {}
        for pair in self.triangle_pairs:
            # In real implementation, this would fetch the current market price
            # Here we'll use mock prices for illustration
            if pair in self.orderbook:
                bid = self.orderbook[pair].get("bids", [])
                ask = self.orderbook[pair].get("asks", [])
                if bid and ask:
                    triangle_prices[pair] = {
                        "bid": bid[0][0],  # Best bid price
                        "ask": ask[0][0],  # Best ask price
                    }
        
        # Skip if we don't have prices for all pairs
        if len(triangle_prices) < len(self.triangle_pairs):
            return indicators
        
        # Calculate triangular arbitrage rate
        # For a triangle A->B->C->A:
        # 1. Convert A to B
        # 2. Convert B to C
        # 3. Convert C back to A
        # If final amount > initial amount, there's a profit opportunity
        
        # Mock calculation (simplified)
        initial_amount = 100.0  # Base currency units
        
        # Convert through the triangle (simplified example)
        # A->B: Sell base for first coin
        pair1 = self.triangle_pairs[0]
        pair1_rate = triangle_prices[pair1]["bid"]  # Use bid price when selling
        intermediate1_amount = initial_amount * pair1_rate
        
        # B->C: Sell first coin for second coin
        pair2 = self.triangle_pairs[1]
        pair2_rate = triangle_prices[pair2]["bid"]
        intermediate2_amount = intermediate1_amount * pair2_rate
        
        # C->A: Sell second coin back to base
        pair3 = self.triangle_pairs[2]
        pair3_rate = triangle_prices[pair3]["bid"]
        final_amount = intermediate2_amount * pair3_rate
        
        # Calculate profit percentage
        gross_profit_pct = (final_amount - initial_amount) / initial_amount
        
        # Adjust for fees (3 trades)
        fee_rate = self.session.trading_fees
        fee_adjusted_profit = gross_profit_pct - (3 * fee_rate)
        
        triangle_opportunity = {
            "triangle": "->".join(self.triangle_pairs),
            "initial_amount": initial_amount,
            "final_amount": final_amount,
            "gross_profit_pct": gross_profit_pct,
            "fee_adjusted_profit": fee_adjusted_profit,
            "profitable": fee_adjusted_profit > self.min_profit_threshold
        }
        
        indicators["triangle_rates"].append(triangle_opportunity)
        indicators["max_profit_pct"] = fee_adjusted_profit
        
        if fee_adjusted_profit > self.min_profit_threshold:
            indicators["best_opportunity"] = triangle_opportunity
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate arbitrage signals based on price differentials or triangle rates.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated arbitrage indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            "arbitrage_opportunities": [],
            "execute_arbitrage": False,
            "arbitrage_type": self.arbitrage_type,
        }
        
        # Process different arbitrage types
        if self.arbitrage_type == "cross_exchange":
            if "best_opportunity" in indicators and indicators["best_opportunity"] is not None:
                best_opp = indicators["best_opportunity"]
                
                # Check if opportunity exceeds threshold
                if best_opp["fee_adjusted_spread"] > self.min_profit_threshold:
                    signals["execute_arbitrage"] = True
                    signals["arbitrage_opportunities"].append({
                        "buy_exchange": best_opp["buy_exchange"],
                        "sell_exchange": best_opp["sell_exchange"],
                        "buy_price": best_opp["buy_price"],
                        "sell_price": best_opp["sell_price"],
                        "expected_profit_pct": best_opp["fee_adjusted_spread"],
                        "timestamp": datetime.utcnow()
                    })
                    
                    logger.info(f"Found cross-exchange arbitrage opportunity: "
                               f"Buy on {best_opp['buy_exchange']} at {best_opp['buy_price']:.2f}, "
                               f"Sell on {best_opp['sell_exchange']} at {best_opp['sell_price']:.2f}, "
                               f"Expected profit: {best_opp['fee_adjusted_spread']*100:.4f}%")
        
        else:  # triangular
            if "best_opportunity" in indicators and indicators["best_opportunity"] is not None:
                best_opp = indicators["best_opportunity"]
                
                # Check if opportunity exceeds threshold
                if best_opp["fee_adjusted_profit"] > self.min_profit_threshold:
                    signals["execute_arbitrage"] = True
                    signals["arbitrage_opportunities"].append({
                        "triangle": best_opp["triangle"],
                        "initial_amount": best_opp["initial_amount"],
                        "expected_final_amount": best_opp["final_amount"],
                        "expected_profit_pct": best_opp["fee_adjusted_profit"],
                        "timestamp": datetime.utcnow()
                    })
                    
                    logger.info(f"Found triangular arbitrage opportunity: "
                               f"Triangle: {best_opp['triangle']}, "
                               f"Expected profit: {best_opp['fee_adjusted_profit']*100:.4f}%")
        
        # Add general market indicators
        signals["market_indicators"] = {
            "current_price": data["close"].iloc[-1] if not data.empty else 0,
            "volume": data["volume"].iloc[-1] if not data.empty else 0,
        }
        
        return signals
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size for arbitrage trades.
        
        For arbitrage, position sizing is primarily based on:
        1. Available liquidity on both exchanges
        2. Risk limits per exchange
        3. Spread/profit size (larger spreads can justify larger positions)
        
        Args:
            direction: Direction of the trade (not as relevant for arbitrage)
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size in crypto units
        """
        # Base position size on account balance and risk parameters
        account_balance = 10000.0  # Mock value, would come from exchange API
        position_size_pct = self.parameters.get("position_size_pct", 0.05)
        
        # Calculate base position size
        base_position_size = account_balance * position_size_pct
        
        # Adjust based on profit opportunity
        profit_threshold = self.min_profit_threshold
        
        max_profit = 0.0
        if self.arbitrage_type == "cross_exchange" and "best_opportunity" in indicators and indicators["best_opportunity"]:
            max_profit = indicators["best_opportunity"]["fee_adjusted_spread"]
        elif self.arbitrage_type == "triangular" and "best_opportunity" in indicators and indicators["best_opportunity"]:
            max_profit = indicators["best_opportunity"]["fee_adjusted_profit"]
        
        # Scale position size based on profit margin (larger profits = larger positions)
        profit_ratio = max_profit / profit_threshold if profit_threshold > 0 else 1.0
        adjusted_position_size = base_position_size * min(profit_ratio, 2.0)  # Cap at 2x
        
        # Ensure minimum trade size
        min_trade_size = self.session.min_trade_size
        current_price = data["close"].iloc[-1] if not data.empty else 0
        
        # Convert to crypto units
        if current_price > 0:
            position_size_in_crypto = adjusted_position_size / current_price
            
            # Respect minimum trade size
            position_size_in_crypto = max(position_size_in_crypto, min_trade_size)
            
            # Round to appropriate precision based on asset
            decimals = 8 if self.session.symbol.startswith("BTC") else 6
            position_size_in_crypto = round(position_size_in_crypto, decimals)
            
            return position_size_in_crypto
        
        return 0.0
    
    def _on_orderbook_updated(self, event: Event) -> None:
        """
        Handle orderbook updated events.
        
        Arbitrage strategies rely heavily on orderbook data from multiple exchanges.
        """
        super()._on_orderbook_updated(event)
        
        # Extract symbol and exchange from event
        symbol = event.data.get('symbol')
        exchange = event.data.get('exchange')
        
        # For cross-exchange arbitrage, update price data for the exchange
        if self.arbitrage_type == "cross_exchange" and exchange in self.exchanges:
            orderbook = event.data.get('orderbook', {})
            
            # Extract mid price from orderbook
            if orderbook and 'bids' in orderbook and 'asks' in orderbook:
                if orderbook['bids'] and orderbook['asks']:
                    best_bid = orderbook['bids'][0][0]
                    best_ask = orderbook['asks'][0][0]
                    mid_price = (best_bid + best_ask) / 2
                    
                    # Update price for this exchange
                    self.exchange_prices[exchange] = {
                        "price": mid_price,
                        "timestamp": datetime.utcnow()
                    }
                    
        # Check for arbitrage opportunities
        if self.is_active:
            # Calculate indicators based on updated prices
            indicators = self.calculate_indicators(self.market_data)
            
            # Generate signals based on indicators
            signals = self.generate_signals(self.market_data, indicators)
            
            # Check for trade opportunities
            if signals.get("execute_arbitrage", False):
                self._check_for_trade_opportunities()
    
    def _on_market_data_updated(self, event: Event) -> None:
        """
        Handle market data updated events.
        
        For arbitrage, we're primarily interested in the order book rather than OHLCV data,
        but we still process market data for analysis and record-keeping.
        """
        super()._on_market_data_updated(event)
        
        # Extract exchange if available
        exchange = event.data.get('exchange')
        
        # For cross-exchange arbitrage, update price for this exchange
        if self.arbitrage_type == "cross_exchange" and exchange in self.exchanges:
            # Extract latest price
            if 'close' in event.data:
                price = event.data['close']
                
                # Update price for this exchange
                self.exchange_prices[exchange] = {
                    "price": price,
                    "timestamp": datetime.utcnow()
                }
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible this strategy is with the current market regime.
        
        Arbitrage strategies can work in virtually any market regime as they're market-neutral,
        but they tend to do better in certain conditions.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            "ranging": 0.85,        # Good in stable, ranging markets
            "volatile": 0.75,       # Can work well during volatility (wider spreads)
            "trending": 0.70,       # Works in trending markets too
            "calm": 0.80,           # Stable markets are good for arbitrage
            "breakout": 0.60,       # Can be affected during breakouts (execution risk)
            "high_volume": 0.90,    # Excellent during high volume (easier execution)
            "low_volume": 0.40,     # Poor during low volume (execution risk)
            "high_liquidity": 0.95, # Excellent in high liquidity markets
            "low_liquidity": 0.30,  # Very poor in low liquidity markets
        }
        
        return compatibility_map.get(market_regime, 0.70)  # Default compatibility is decent
