from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT = "EXIT"

class Strategy:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.last_signal: Optional[SignalType] = None
        self.last_signal_time: Optional[datetime] = None
        
    def generate_signal(self, data: Dict[str, Any]) -> SignalType:
        """
        Generate trading signal based on strategy logic and market data
        Must be implemented by concrete strategy classes
        """
        raise NotImplementedError("Strategy must implement generate_signal method")
    
    def update(self, data: Dict[str, Any]) -> None:
        """
        Update strategy state with new market data
        Optional method that can be overridden by concrete strategies
        """
        pass
    
    def reset(self) -> None:
        """
        Reset strategy state
        Optional method that can be overridden by concrete strategies
        """
        self.last_signal = None
        self.last_signal_time = None 