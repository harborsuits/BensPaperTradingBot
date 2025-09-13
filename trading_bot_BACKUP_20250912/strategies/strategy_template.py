#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Template Module

This is a compatibility module to re-export the Strategy Template from its actual location.
"""

# Re-export all components from the strategy_template
from trading_bot.strategies.factory.strategy_template import *

"""
Strategy template module defining common signal types and structures.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

class SignalType(Enum):
    """Enum representing different types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT = "exit"
    NEUTRAL = "neutral"

@dataclass
class Signal:
    """Class representing a trading signal."""
    type: SignalType
    timestamp: datetime
    symbol: str
    price: float
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary format."""
        return {
            'type': self.type.value,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'price': self.price,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Create signal from dictionary format."""
        return cls(
            type=SignalType(data['type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            symbol=data['symbol'],
            price=data['price'],
            confidence=data['confidence'],
            metadata=data['metadata']
        )
        
    def __str__(self) -> str:
        """String representation of signal."""
        return (f"{self.type.value.upper()} {self.symbol} @ {self.price:.2f} "
                f"(Confidence: {self.confidence:.2f})")
