#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Dollar-Cost Averaging (DCA) Strategy

This strategy implements the popular dollar-cost averaging investment approach
for cryptocurrencies. It makes regular, fixed-size purchases at scheduled intervals
regardless of price, with optional enhancements for market condition awareness.

Key characteristics:
- Scheduled, automated purchases
- Fixed or dynamic investment amounts
- Long-term investment horizon
- Reduced emotional decision-making
- Optional market condition adjustments
"""

import logging
from typing import Dict, Any, List, Optional
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
    name="CryptoDCAStrategy",
    market_type="crypto",
    description="Dollar-Cost Averaging strategy for long-term crypto investment with scheduled purchases",
    timeframes=["D1", "W1"],  # DCA typically operates on daily or weekly intervals
    parameters={
        # Schedule parameters
        "frequency": {"type": "str", "default": "weekly", "enum": ["daily", "weekly", "biweekly", "monthly"]},
        "day_of_week": {"type": "int", "default": 1, "min": 1, "max": 7},  # 1=Monday, 7=Sunday
        "day_of_month": {"type": "int", "default": 1, "min": 1, "max": 28},
        "hour_of_day": {"type": "int", "default": 0, "min": 0, "max": 23},
        
        # Investment parameters
        "base_investment_amount": {"type": "float", "default": 100.0, "min": 10.0, "max": 10000.0},
        "dynamic_sizing": {"type": "bool", "default": False},
        
        # Optional enhancements
        "value_averaging_enabled": {"type": "bool", "default": False},
        "target_allocation_pct": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0},
        "adjust_for_volatility": {"type": "bool", "default": False},
        "increase_in_downtrends": {"type": "bool", "default": False},
        "dip_buying_enabled": {"type": "bool", "default": False},
        "dip_threshold_pct": {"type": "float", "default": 20.0, "min": 5.0, "max": 50.0},
        
        # Portfolio parameters
        "max_portfolio_value_pct": {"type": "float", "default": 100.0, "min": 1.0, "max": 100.0},
        "reinvest_profits": {"type": "bool", "default": False},
    }
)
class CryptoDCAStrategy(CryptoBaseStrategy):
    """
    A Dollar-Cost Averaging strategy for cryptocurrency investment.
    
    This strategy:
    1. Makes regular purchases at fixed intervals (daily, weekly, biweekly, or monthly)
    2. Invests a predetermined amount each time, regardless of price
    3. Can optionally adjust purchase amounts based on market conditions
    4. Focuses on long-term accumulation rather than short-term trading
    5. Aims to reduce the impact of volatility and emotional decision-making
    """
    
    def __init__(self, session: CryptoSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """Initialize the crypto DCA strategy."""
        super().__init__(session, data_pipeline, parameters)
        
        # DCA-specific state
        self.last_purchase_time = None
        self.next_scheduled_purchase = None
        self.purchase_history = []
        self.total_invested = 0.0
        self.total_purchased = 0.0
        self.average_purchase_price = 0.0
        
        # Portfolio tracking
        self.current_portfolio_value = 0.0
        self.target_portfolio_value = 0.0
        
        # Schedule the first purchase
        self._schedule_next_purchase()
        
        logger.info(f"Initialized crypto DCA strategy for {self.session.symbol} with {self.parameters['frequency']} purchases")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators for DCA strategy.
        
        For basic DCA, we don't need many indicators since purchases are made
        regardless of price. However, for enhanced versions, we calculate
        some market condition indicators.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if data.empty:
            return indicators
        
        # Basic market statistics
        indicators["current_price"] = data["close"].iloc[-1]
        
        # Only calculate additional indicators if we're using enhanced DCA
        if self._using_enhanced_dca():
            # Calculate Simple Moving Averages for trend detection
            indicators["sma_50"] = data["close"].rolling(window=50).mean()
            indicators["sma_200"] = data["close"].rolling(window=200).mean()
            
            # Market trend detection
            indicators["market_trend"] = "neutral"
            if len(indicators["sma_50"]) > 0 and len(indicators["sma_200"]) > 0:
                if indicators["sma_50"].iloc[-1] > indicators["sma_200"].iloc[-1]:
                    indicators["market_trend"] = "bullish"
                elif indicators["sma_50"].iloc[-1] < indicators["sma_200"].iloc[-1]:
                    indicators["market_trend"] = "bearish"
            
            # Volatility measurement
            if len(data) >= 20:
                price_returns = data["close"].pct_change()
                indicators["volatility"] = price_returns.iloc[-20:].std() * np.sqrt(365)  # Annualized
            
            # Dip detection (for dip buying enhancement)
            if self.parameters["dip_buying_enabled"] and len(data) >= 30:
                # Define a dip as X% below the 30-day high
                thirty_day_high = data["high"].iloc[-30:].max()
                current_price = data["close"].iloc[-1]
                dip_amount_pct = (thirty_day_high - current_price) / thirty_day_high * 100
                indicators["current_dip_pct"] = dip_amount_pct
                indicators["in_dip"] = dip_amount_pct >= self.parameters["dip_threshold_pct"]
                
            # Value averaging calculations (if enabled)
            if self.parameters["value_averaging_enabled"]:
                indicators["target_portfolio_value"] = self._calculate_target_portfolio_value()
                indicators["value_gap"] = indicators["target_portfolio_value"] - self.current_portfolio_value
        
        return indicators
    
    def _using_enhanced_dca(self) -> bool:
        """Check if we're using any enhanced DCA features."""
        return (
            self.parameters["dynamic_sizing"] or 
            self.parameters["value_averaging_enabled"] or
            self.parameters["adjust_for_volatility"] or
            self.parameters["increase_in_downtrends"] or
            self.parameters["dip_buying_enabled"]
        )
    
    def _calculate_target_portfolio_value(self) -> float:
        """
        Calculate the target portfolio value for value averaging.
        
        For value averaging, the portfolio should grow by a fixed amount each period.
        """
        if not self.last_purchase_time:
            return self.parameters["base_investment_amount"]
            
        # Calculate how many periods have passed since beginning
        start_date = self.purchase_history[0]["date"] if self.purchase_history else datetime.utcnow()
        current_date = datetime.utcnow()
        
        # Convert frequency to days for calculation
        frequency_days = {
            "daily": 1,
            "weekly": 7,
            "biweekly": 14,
            "monthly": 30
        }
        days_per_period = frequency_days.get(self.parameters["frequency"], 7)
        
        # Calculate periods since start
        days_since_start = (current_date - start_date).days
        periods_since_start = max(1, days_since_start // days_per_period)
        
        # Target value increases by base amount each period
        return periods_since_start * self.parameters["base_investment_amount"]
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate buying signals for DCA strategy.
        
        For DCA, signals are primarily based on the schedule rather than
        technical indicators. However, for enhanced DCA, market conditions
        may influence purchase size.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            "dca_purchase": False,
            "purchase_amount": 0.0,
            "purchase_type": "scheduled",
            "schedule_info": {},
        }
        
        # Get current time
        current_time = datetime.utcnow()
        
        # Update portfolio value
        if "current_price" in indicators and self.total_purchased > 0:
            self.current_portfolio_value = self.total_purchased * indicators["current_price"]
        
        # Check if it's time for scheduled purchase
        if self.next_scheduled_purchase and current_time >= self.next_scheduled_purchase:
            signals["dca_purchase"] = True
            signals["purchase_type"] = "scheduled"
            signals["schedule_info"] = {
                "scheduled_time": self.next_scheduled_purchase,
                "frequency": self.parameters["frequency"],
            }
            
            # Calculate base purchase amount
            purchase_amount = self.parameters["base_investment_amount"]
            
            # Apply enhancements if enabled
            if self._using_enhanced_dca():
                purchase_amount = self._adjust_purchase_amount(purchase_amount, indicators)
            
            signals["purchase_amount"] = purchase_amount
            
            logger.info(f"DCA signal generated for {self.session.symbol} - {signals['purchase_type']}")
        
        # Special case: check for dip buying if enabled
        elif self.parameters["dip_buying_enabled"] and indicators.get("in_dip", False):
            dip_pct = indicators.get("current_dip_pct", 0)
            
            # Only trigger if we haven't bought in the last day (to prevent multiple dip buys)
            if not self.last_purchase_time or (current_time - self.last_purchase_time).days >= 1:
                signals["dca_purchase"] = True
                signals["purchase_type"] = "dip_buying"
                
                # Calculate dip purchase amount (usually base amount, can be adjusted)
                base_amount = self.parameters["base_investment_amount"]
                
                # Optionally scale by dip size
                dip_scale_factor = 1.0 + (dip_pct / 100)  # E.g., 20% dip = 1.2x scaling
                purchase_amount = base_amount * min(dip_scale_factor, 2.0)  # Cap at 2x
                
                signals["purchase_amount"] = purchase_amount
                signals["dip_percentage"] = dip_pct
                
                logger.info(f"DCA dip buying signal for {self.session.symbol} - {dip_pct:.1f}% dip")
        
        return signals
    
    def _adjust_purchase_amount(self, base_amount: float, indicators: Dict[str, Any]) -> float:
        """
        Adjust the purchase amount based on market conditions for enhanced DCA.
        
        Args:
            base_amount: Base purchase amount
            indicators: Market indicators
            
        Returns:
            Adjusted purchase amount
        """
        adjusted_amount = base_amount
        
        # Value averaging adjustment
        if self.parameters["value_averaging_enabled"] and "value_gap" in indicators:
            value_gap = indicators["value_gap"]
            if value_gap > 0:
                # Portfolio is below target, increase purchase
                adjusted_amount = value_gap
            else:
                # Portfolio is above target, decrease or skip purchase
                adjusted_amount = max(0, base_amount / 2)  # Never sell, just reduce purchases
            
            logger.info(f"Value averaging adjustment: {base_amount:.2f} → {adjusted_amount:.2f}")
        
        # Adjust for market trend
        elif self.parameters["increase_in_downtrends"] and "market_trend" in indicators:
            if indicators["market_trend"] == "bearish":
                # Increase purchases in downtrends
                adjusted_amount = base_amount * 1.5
                logger.info(f"Increasing purchase in downtrend: {base_amount:.2f} → {adjusted_amount:.2f}")
        
        # Adjust for volatility
        if self.parameters["adjust_for_volatility"] and "volatility" in indicators:
            volatility = indicators["volatility"]
            avg_volatility = 0.7  # Typical crypto volatility
            
            if volatility > avg_volatility:
                # Higher volatility = more DCA benefit, increase slightly
                vol_factor = min(volatility / avg_volatility, 1.5)  # Cap at 1.5x
                adjusted_amount = adjusted_amount * vol_factor
                logger.info(f"Volatility adjustment: factor {vol_factor:.2f}")
        
        return adjusted_amount
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size for DCA purchase.
        
        For DCA, the position size is determined by the purchase amount and current price.
        
        Args:
            direction: Direction (always "long" for DCA)
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size in crypto units
        """
        if data.empty or "current_price" not in indicators:
            return 0.0
        
        # For DCA, we only open long positions
        if direction != "long":
            return 0.0
        
        # Get purchase amount from signals
        purchase_amount = 0.0
        if self.signals and "purchase_amount" in self.signals:
            purchase_amount = self.signals["purchase_amount"]
        else:
            purchase_amount = self.parameters["base_investment_amount"]
        
        # Calculate crypto amount based on current price
        current_price = indicators["current_price"]
        if current_price > 0:
            position_size_crypto = purchase_amount / current_price
            
            # Ensure minimum trade size
            min_trade_size = self.session.min_trade_size
            position_size_crypto = max(position_size_crypto, min_trade_size)
            
            # Round to appropriate precision
            decimals = 8 if self.session.symbol.startswith("BTC") else 6
            position_size_crypto = round(position_size_crypto, decimals)
            
            return position_size_crypto
        
        return 0.0
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed events.
        
        For DCA, we check if it's time for a scheduled purchase.
        """
        super()._on_timeframe_completed(event)
        
        # Only process if this is our timeframe
        if event.data.get('timeframe') != self.session.timeframe:
            return
        
        # Calculate indicators
        self.indicators = self.calculate_indicators(self.market_data)
        
        # Generate signals
        self.signals = self.generate_signals(self.market_data, self.indicators)
        
        # Execute if DCA purchase signal is generated
        if self.signals.get("dca_purchase", False):
            self._execute_dca_purchase()
    
    def _execute_dca_purchase(self) -> None:
        """Execute a DCA purchase."""
        # Check for trade opportunities (which will use our signals)
        self._check_for_trade_opportunities()
        
        # Update purchase history and stats
        if self.signals and self.signals.get("dca_purchase", False):
            current_time = datetime.utcnow()
            current_price = self.indicators.get("current_price", 0)
            
            # Record the purchase
            purchase_record = {
                "date": current_time,
                "price": current_price,
                "amount_fiat": self.signals.get("purchase_amount", 0),
                "amount_crypto": 0,  # Will be filled when position is opened
                "type": self.signals.get("purchase_type", "scheduled")
            }
            
            # Add to history
            self.purchase_history.append(purchase_record)
            
            # Update last purchase time
            self.last_purchase_time = current_time
            
            # Schedule next purchase
            self._schedule_next_purchase()
            
            logger.info(f"DCA purchase executed for {self.session.symbol} at {current_price}")
    
    def _on_position_opened(self, event: Event) -> None:
        """Handle position opened events for DCA strategy."""
        super()._on_position_opened(event)
        
        # Update purchase history with the actual crypto amount
        if self.purchase_history and "amount_crypto" in event.data:
            # Update the latest purchase record
            self.purchase_history[-1]["amount_crypto"] = event.data["amount_crypto"]
            
            # Update totals
            self.total_invested += self.purchase_history[-1]["amount_fiat"]
            self.total_purchased += event.data["amount_crypto"]
            
            # Calculate average purchase price
            if self.total_purchased > 0:
                self.average_purchase_price = self.total_invested / self.total_purchased
    
    def _schedule_next_purchase(self) -> None:
        """
        Schedule the next DCA purchase based on frequency settings.
        """
        current_time = datetime.utcnow()
        frequency = self.parameters["frequency"]
        
        if frequency == "daily":
            # Set to the specified hour the next day
            next_date = current_time.date() + timedelta(days=1)
            next_time = datetime.combine(next_date, datetime.min.time()) + timedelta(hours=self.parameters["hour_of_day"])
            
        elif frequency == "weekly":
            # Set to the specified day of week and hour
            days_ahead = self.parameters["day_of_week"] - current_time.isoweekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
                
            next_date = current_time.date() + timedelta(days=days_ahead)
            next_time = datetime.combine(next_date, datetime.min.time()) + timedelta(hours=self.parameters["hour_of_day"])
            
        elif frequency == "biweekly":
            # Set to every other week on specified day
            days_ahead = self.parameters["day_of_week"] - current_time.isoweekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
                
            # Add another week if we're in the off-week
            if self.last_purchase_time:
                days_since_last = (current_time.date() - self.last_purchase_time.date()).days
                if days_since_last < 10:  # Less than 10 days since last purchase, move to next biweekly period
                    days_ahead += 7
                    
            next_date = current_time.date() + timedelta(days=days_ahead)
            next_time = datetime.combine(next_date, datetime.min.time()) + timedelta(hours=self.parameters["hour_of_day"])
            
        elif frequency == "monthly":
            # Set to the specified day of month and hour
            day = min(self.parameters["day_of_month"], 28)  # Avoid month boundary issues
            
            # Get current month's target date
            year, month = current_time.year, current_time.month
            next_date = datetime(year, month, day).date()
            
            # If that date has passed, move to next month
            if next_date <= current_time.date():
                month += 1
                if month > 12:
                    month = 1
                    year += 1
                next_date = datetime(year, month, day).date()
                
            next_time = datetime.combine(next_date, datetime.min.time()) + timedelta(hours=self.parameters["hour_of_day"])
        
        else:
            # Default to weekly if invalid frequency
            next_time = current_time + timedelta(days=7)
        
        self.next_scheduled_purchase = next_time
        logger.info(f"Next DCA purchase for {self.session.symbol} scheduled for {next_time}")
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible this strategy is with the current market regime.
        
        DCA is designed to work in all market regimes, though it provides the most
        benefit during ranging and bearish markets.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            "trending": 0.75,        # Good in trending markets
            "ranging": 0.90,         # Very good in ranging markets
            "volatile": 0.95,        # Excellent in volatile markets
            "calm": 0.60,            # Moderate in calm markets
            "breakout": 0.70,        # Good during breakouts
            "high_volume": 0.80,     # Very good during high volume
            "low_volume": 0.80,      # Very good during low volume
            "high_liquidity": 0.85,  # Very good in high liquidity markets
            "low_liquidity": 0.75,   # Good in low liquidity markets
        }
        
        return compatibility_map.get(market_regime, 0.85)  # Default compatibility is high
