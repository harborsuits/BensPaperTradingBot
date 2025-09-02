"""
Strategy settings classes for the trading bot.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

@dataclass
class StrategySettings:
    """Settings for strategy configuration."""
    
    # Strategy type and identifier
    strategy_type: str = "momentum"  # Default to momentum
    strategy_id: str = "default_strategy"
    
    # Risk parameters
    risk_level: int = 2  # 1-5 scale
    max_position_size_pct: float = 10.0  # As percentage of capital
    
    # Trade management
    use_stop_loss: bool = True
    stop_loss_pct: float = 5.0
    use_take_profit: bool = True  
    take_profit_pct: float = 15.0
    
    # Trading hours/timing
    trading_hours: List[str] = field(default_factory=lambda: ["Market Open"])
    
    # Custom parameters based on strategy type
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default parameters based on strategy type."""
        if not self.parameters:
            if self.strategy_type == "momentum":
                self.parameters = {
                    "lookback_period": 14,
                    "momentum_threshold": 0.05,
                    "trend_filter": True
                }
            elif self.strategy_type == "mean_reversion":
                self.parameters = {
                    "lookback_period": 20,
                    "std_dev_threshold": 2.0,
                    "mean_period": 50
                }
            elif self.strategy_type == "trend_following":
                self.parameters = {
                    "fast_period": 20,
                    "slow_period": 50,
                    "signal_threshold": 0.01
                }
            elif self.strategy_type == "volatility_breakout":
                self.parameters = {
                    "atr_period": 14,
                    "breakout_factor": 1.5,
                    "confirmation_candles": 2
                }


@dataclass
class TradingBotSettings:
    """Overall settings for the trading bot."""
    
    # General settings
    bot_id: str = "main_trading_bot"
    trading_enabled: bool = False
    
    # Capital settings
    initial_capital: float = 10000.0
    max_capital_per_trade: float = 1000.0
    
    # Risk settings
    max_open_positions: int = 5
    max_capital_at_risk_pct: float = 50.0
    
    # Strategy allocation
    strategies: List[StrategySettings] = field(default_factory=list)
    strategy_allocations: Dict[str, float] = field(default_factory=dict)
    
    # Trading schedule
    trading_days: List[str] = field(default_factory=lambda: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    auto_restart: bool = True
    
    # Data settings
    data_provider: str = "yahoo"
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN"])
    
    def __post_init__(self):
        """Initialize with a default strategy if none provided."""
        if not self.strategies:
            self.strategies = [StrategySettings()]
        
        if not self.strategy_allocations:
            # Distribute allocation evenly
            allocation_pct = 100.0 / len(self.strategies)
            self.strategy_allocations = {
                strategy.strategy_id: allocation_pct 
                for strategy in self.strategies
            }
