#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breakout Trading Strategy

This module implements a stock breakout trading strategy that aims to profit from price
movements beyond established support, resistance, or consolidation patterns. The strategy 
is account-aware, ensuring it complies with account balance requirements, regulatory 
constraints, and risk management.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from trading_bot.strategies_new.stocks.base.stocks_base_strategy import StocksBaseStrategy, StocksSession
from trading_bot.strategies_new.validation.account_aware_mixin import AccountAwareMixin
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.core.position import Position, PositionStatus
from trading_bot.strategies_new.factory.registry import register_strategy

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="BreakoutStrategy",
    market_type="stocks",
    description="A strategy that trades breakouts from established price patterns, consolidations, or technical levels",
    timeframes=["1d", "4h", "1h", "30m"],
    parameters={
        "lookback_period": {"description": "Period for identifying support/resistance levels", "type": "int"},
        "breakout_confirmation_bars": {"description": "Number of bars to confirm breakout", "type": "int"},
        "volume_multiplier": {"description": "Minimum volume multiplier for valid breakout", "type": "float"},
        "atr_multiplier": {"description": "ATR multiplier for stop loss calculation", "type": "float"}
    }
)
class BreakoutStrategy(StocksBaseStrategy, AccountAwareMixin):
    """
    Stock Breakout Trading Strategy
    
    This strategy:
    1. Identifies potential breakout levels (support, resistance, consolidation patterns)
    2. Detects and confirms breakouts with price action and volume confirmations
    3. Places entries in the direction of the breakout with proper risk management
    4. Uses trailing stops to ride the momentum of successful breakouts
    5. Incorporates account awareness for regulatory compliance and risk management
    
    The breakout strategy works best in trending markets after periods of consolidation
    and is designed for short to medium-term trades.
    """
    
    def __init__(self, session: StocksSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the Breakout Trading strategy.
        
        Args:
            session: Stock trading session with symbol, timeframe, etc.
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Initialize parent classes
        StocksBaseStrategy.__init__(self, session, data_pipeline, parameters)
        AccountAwareMixin.__init__(self)
        
        # Default parameters for breakout trading
        default_params = {
            # Strategy identification
            'strategy_name': 'Breakout Trading',
            'strategy_id': 'breakout_trading',
            'is_day_trade': False,  # Can be intraday or multi-day
            
            # Breakout identification parameters
            'lookback_period': 20,
            'breakout_confirmation_bars': 2,
            'consolidation_threshold': 0.03,  # Max 3% range for consolidation pattern
            'min_consolidation_bars': 5,  # Minimum bars in consolidation
            
            # Support/Resistance parameters
            'min_touches': 2,  # Minimum number of touches for level validation
            'level_tolerance': 0.005,  # 0.5% tolerance for level identification
            
            # Volume and momentum confirmation
            'volume_multiplier': 1.5,  # Breakout volume should be 1.5x average
            'macd_confirmation': True,  # Use MACD for trend confirmation
            
            # Trade execution
            'max_positions': 3,
            'atr_period': 14,
            'atr_multiplier': 2.0,  # ATR multiplier for stop loss
            'trailing_stop': True,
            'trailing_stop_activation': 0.02,  # Activate trailing stop after 2% profit
            'trailing_stop_distance': 0.03,  # 3% trailing stop
            
            # Risk management
            'risk_per_trade': 0.01,  # 1% risk per trade
            'reward_risk_ratio': 2.0,  # Minimum reward:risk ratio
            'max_position_size_pct': 0.10,  # Maximum 10% of account in a single position
        }
        
        # Update with user-provided parameters
        if parameters:
            default_params.update(parameters)
        self.parameters = default_params
        
        # Strategy state
        self.support_levels = []
        self.resistance_levels = []
        self.consolidation_patterns = []
        self.current_breakout = {
            'type': None,  # 'support', 'resistance', 'consolidation'
            'level': None,
            'direction': None,
            'confirmed': False,
            'detected_at': None,
            'strength': 0.0
        }
        
        logger.info(f"Initialized {self.name} for {session.symbol} on {session.timeframe}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the Breakout Trading strategy.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if data.empty or len(data) < max(self.parameters['lookback_period'], 
                                        self.parameters['atr_period']):
            return indicators
        
        try:
            # Calculate ATR for volatility assessment
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators['atr'] = true_range.rolling(window=self.parameters['atr_period']).mean()
            
            # Calculate Volume metrics
            indicators['volume_sma'] = data['volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = data['volume'] / indicators['volume_sma']
            
            # Calculate MACD if needed for confirmation
            if self.parameters['macd_confirmation']:
                ema12 = data['close'].ewm(span=12, adjust=False).mean()
                ema26 = data['close'].ewm(span=26, adjust=False).mean()
                indicators['macd'] = ema12 - ema26
                indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
                indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # Identify key levels and patterns
            self._identify_key_levels(data)
            self._identify_consolidation_patterns(data)
            
            # Detect potential breakouts
            self._detect_breakouts(data, indicators)
            
            # Calculate additional breakout metrics if a breakout is detected
            if self.current_breakout['level'] is not None:
                current_price = data['close'].iloc[-1]
                breakout_level = self.current_breakout['level']
                
                # Calculate distance from breakout level
                indicators['breakout_distance'] = (current_price - breakout_level) / breakout_level
                
                # Calculate breakout momentum
                indicators['breakout_momentum'] = data['close'].iloc[-1] - data['close'].iloc[-5] if len(data) >= 5 else 0
                
        except Exception as e:
            logger.error(f"Error calculating breakout indicators: {str(e)}")
        
        return indicators
