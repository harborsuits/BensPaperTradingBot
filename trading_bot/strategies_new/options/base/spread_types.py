#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spread Types Module

This module defines the various types of option spreads supported by the vertical
spread engine, along with helper classes to represent spread positions and properties.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, date

class OptionType(Enum):
    """Type of option (call or put)."""
    CALL = "call"
    PUT = "put"

class VerticalSpreadType(Enum):
    """Types of vertical spreads supported by the engine."""
    BULL_CALL_SPREAD = "bull_call_spread"  # Debit spread (long lower strike call, short higher strike call)
    BEAR_CALL_SPREAD = "bear_call_spread"  # Credit spread (short lower strike call, long higher strike call)
    BULL_PUT_SPREAD = "bull_put_spread"    # Credit spread (short higher strike put, long lower strike put)
    BEAR_PUT_SPREAD = "bear_put_spread"    # Debit spread (long higher strike put, short lower strike put)
    
    @classmethod
    def is_bullish(cls, spread_type):
        """Check if a spread type is bullish."""
        return spread_type in [cls.BULL_CALL_SPREAD, cls.BULL_PUT_SPREAD]
    
    @classmethod
    def is_bearish(cls, spread_type):
        """Check if a spread type is bearish."""
        return spread_type in [cls.BEAR_CALL_SPREAD, cls.BEAR_PUT_SPREAD]
    
    @classmethod
    def is_credit(cls, spread_type):
        """Check if a spread type is a credit spread."""
        return spread_type in [cls.BEAR_CALL_SPREAD, cls.BULL_PUT_SPREAD]
    
    @classmethod
    def is_debit(cls, spread_type):
        """Check if a spread type is a debit spread."""
        return spread_type in [cls.BULL_CALL_SPREAD, cls.BEAR_PUT_SPREAD]
    
    @classmethod
    def get_option_type(cls, spread_type):
        """Get the option type used in this spread."""
        if spread_type in [cls.BULL_CALL_SPREAD, cls.BEAR_CALL_SPREAD]:
            return OptionType.CALL
        else:
            return OptionType.PUT

@dataclass
class OptionContract:
    """Represents a single option contract."""
    symbol: str
    option_type: OptionType
    strike: float
    expiration: date
    bid: float
    ask: float
    last: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    implied_volatility: Optional[float] = None
    
    @property
    def mid_price(self) -> float:
        """Calculate mid-price between bid and ask."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def spread_percentage(self) -> float:
        """Calculate bid-ask spread as percentage of ask price."""
        if self.ask == 0:
            return float('inf')
        return (self.ask - self.bid) / self.ask

@dataclass
class VerticalSpread:
    """Represents a vertical spread position."""
    spread_type: VerticalSpreadType
    long_option: OptionContract
    short_option: OptionContract
    quantity: int = 1
    entry_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    
    @property
    def width(self) -> float:
        """Get the width of the spread (distance between strikes)."""
        return abs(self.long_option.strike - self.short_option.strike)
    
    @property
    def net_premium(self) -> float:
        """
        Calculate the net premium of the spread.
        
        For debit spreads, this is a positive value.
        For credit spreads, this is a negative value.
        """
        # For call spreads
        if self.long_option.option_type == OptionType.CALL:
            if self.long_option.strike < self.short_option.strike:  # Bull call spread
                return self.long_option.mid_price - self.short_option.mid_price
            else:  # Bear call spread
                return self.short_option.mid_price - self.long_option.mid_price
        # For put spreads
        else:
            if self.long_option.strike < self.short_option.strike:  # Bull put spread
                return self.short_option.mid_price - self.long_option.mid_price
            else:  # Bear put spread
                return self.long_option.mid_price - self.short_option.mid_price
    
    @property
    def max_profit(self) -> float:
        """Calculate maximum potential profit for this spread."""
        spread_type = self.spread_type
        
        if VerticalSpreadType.is_credit(spread_type):
            # Credit spreads: max profit is the premium received
            return abs(self.net_premium) * 100 * self.quantity
        else:
            # Debit spreads: max profit is width minus premium paid
            return (self.width - abs(self.net_premium)) * 100 * self.quantity
    
    @property
    def max_loss(self) -> float:
        """Calculate maximum potential loss for this spread."""
        spread_type = self.spread_type
        
        if VerticalSpreadType.is_credit(spread_type):
            # Credit spreads: max loss is width minus premium received
            return (self.width - abs(self.net_premium)) * 100 * self.quantity
        else:
            # Debit spreads: max loss is the premium paid
            return abs(self.net_premium) * 100 * self.quantity
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk to reward ratio."""
        if self.max_profit == 0:
            return float('inf')
        return self.max_loss / self.max_profit
    
    @property
    def breakeven(self) -> float:
        """Calculate breakeven price at expiration."""
        spread_type = self.spread_type
        
        if spread_type == VerticalSpreadType.BULL_CALL_SPREAD:
            return self.long_option.strike + abs(self.net_premium)
        elif spread_type == VerticalSpreadType.BEAR_CALL_SPREAD:
            return self.short_option.strike + abs(self.net_premium)
        elif spread_type == VerticalSpreadType.BULL_PUT_SPREAD:
            return self.short_option.strike - abs(self.net_premium)
        elif spread_type == VerticalSpreadType.BEAR_PUT_SPREAD:
            return self.long_option.strike - abs(self.net_premium)
        
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert spread to dictionary for storage/serialization."""
        return {
            "spread_type": self.spread_type.value,
            "long_option": {
                "symbol": self.long_option.symbol,
                "option_type": self.long_option.option_type.value,
                "strike": self.long_option.strike,
                "expiration": self.long_option.expiration.isoformat(),
                "bid": self.long_option.bid,
                "ask": self.long_option.ask,
            },
            "short_option": {
                "symbol": self.short_option.symbol,
                "option_type": self.short_option.option_type.value,
                "strike": self.short_option.strike,
                "expiration": self.short_option.expiration.isoformat(),
                "bid": self.short_option.bid,
                "ask": self.short_option.ask,
            },
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "width": self.width,
            "net_premium": self.net_premium,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "risk_reward_ratio": self.risk_reward_ratio,
            "breakeven": self.breakeven
        }
