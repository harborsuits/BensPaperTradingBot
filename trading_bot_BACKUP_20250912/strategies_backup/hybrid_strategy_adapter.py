"""
Hybrid Strategy Adapter

This module provides an adapter to integrate the hybrid strategy system
with the existing strategy factory for seamless use throughout the trading bot.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd

from trading_bot.strategies.hybrid_strategy_system import HybridStrategySystem, create_hybrid_strategy_system

logger = logging.getLogger(__name__)

class HybridStrategyAdapter:
    """
    Adapter for the hybrid strategy system
    
    Conforms to the standard strategy interface expected by the trading bot
    while providing advanced hybrid strategy capabilities.
    """
    
    def __init__(self, name="Hybrid Strategy", parameters=None):
        """
        Initialize the hybrid strategy adapter
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
        """
        self.name = name
        self.params = parameters or {}
        
        # Create the underlying hybrid strategy system
        self.hybrid_system = create_hybrid_strategy_system(config=self.params)
        logger.info(f"Initialized {self.name} with HybridStrategySystem")
        
        # Track recent history locally (for UI display)
        self.recent_signals = []
        self.max_history_size = 50
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if strategy is available"""
        return True
    
    def generate_signals(self, data, **kwargs) -> Dict[str, Any]:
        """
        Generate trading signals based on hybrid strategy logic
        
        Args:
            data: DataFrame with OHLCV price data
            **kwargs: Additional parameters including ticker and timeframe
            
        Returns:
            Dict: Signal information
        """
        # Extract ticker and timeframe from kwargs
        ticker = kwargs.get('ticker', None)
        timeframe = kwargs.get('timeframe', '1d')
        
        # Generate signals from the hybrid system
        signal = self.hybrid_system.generate_signals(data, ticker=ticker, timeframe=timeframe)
        
        # Store in local history
        self.recent_signals.append({
            'ticker': ticker,
            'timeframe': timeframe,
            'timestamp': signal.get('timestamp'),
            'action': signal.get('action'),
            'confidence': signal.get('confidence'),
            'explanation': signal.get('explanation'),
            'votes': signal.get('votes')
        })
        
        # Limit history size
        if len(self.recent_signals) > self.max_history_size:
            self.recent_signals = self.recent_signals[-self.max_history_size:]
        
        # Return standard signals expected by trading bot
        return {
            'action': signal.get('action', 'hold'),
            'confidence': signal.get('confidence', 0.0),
            'reason': signal.get('explanation', ''),
            'target': signal.get('risk_params', {}).get('target', None),
            'stop': signal.get('risk_params', {}).get('stop', None),
            'stop_loss_pct': signal.get('risk_params', {}).get('stop_loss_pct', None),
            'take_profit_pct': signal.get('risk_params', {}).get('take_profit_pct', None),
            'hybrid_votes': signal.get('votes', {})
        }
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics for the hybrid strategy system
        
        Returns:
            Dict: Performance metrics
        """
        return self.hybrid_system.get_strategy_performance()
    
    def get_strategy_votes(self) -> List[Dict[str, Any]]:
        """
        Get recent strategy voting history
        
        Returns:
            List: Recent strategy signals with voting info
        """
        return self.recent_signals
    
    def get_component_weights(self) -> Dict[str, float]:
        """
        Get current weights for each strategy component
        
        Returns:
            Dict: Strategy component weights
        """
        return self.hybrid_system.strategy_weights
    
    def set_component_weights(self, weights: Dict[str, float]) -> bool:
        """
        Set new weights for strategy components
        
        Args:
            weights: Dictionary with strategy component weights
            
        Returns:
            bool: Success status
        """
        # Validate weights
        if not isinstance(weights, dict):
            logger.error("Weights must be a dictionary")
            return False
            
        required_keys = ['technical', 'ml', 'weighted_avg_peak']
        if not all(key in weights for key in required_keys):
            logger.error(f"Weights dictionary must contain keys: {required_keys}")
            return False
            
        # Check that weights sum to 1.0 (or close to it with floating point)
        total = sum(weights.values())
        if not (0.99 <= total <= 1.01):
            logger.error(f"Weights must sum to 1.0, got {total}")
            return False
            
        # Set weights
        self.hybrid_system.strategy_weights = weights
        logger.info(f"Updated hybrid strategy weights: {weights}")
        return True
    
    def export_strategy_history(self, filepath: str):
        """
        Export strategy history to file
        
        Args:
            filepath: Path to export file
        """
        self.hybrid_system.export_strategy_history(filepath)
    
    def import_strategy_history(self, filepath: str):
        """
        Import strategy history from file
        
        Args:
            filepath: Path to import file
        """
        self.hybrid_system.import_strategy_history(filepath)
