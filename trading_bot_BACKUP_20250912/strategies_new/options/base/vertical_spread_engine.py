#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vertical Spread Engine

This module provides the core implementation of the vertical spread engine,
combining spread analysis, position management, and signal generation
into a cohesive strategy framework.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.strategies_new.options.base.options_base_strategy import OptionsBaseStrategy, OptionsSession
from trading_bot.strategies_new.options.base.spread_types import (
    OptionType, VerticalSpreadType, OptionContract, VerticalSpread
)
from trading_bot.strategies_new.options.base.spread_analyzer import SpreadAnalyzer
from trading_bot.strategies_new.options.base.spread_manager import SpreadManager, SpreadPosition
from trading_bot.strategies_new.factory.registry import register_strategy

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="VerticalSpreadEngine",
    market_type="options",
    description="Flexible vertical spread strategy engine for options trading",
    timeframes=["D1"],
    parameters={
        # Market bias indicators
        "use_rsi": {"type": "bool", "default": True},
        "rsi_period": {"type": "int", "default": 14, "min": 7, "max": 30},
        "rsi_overbought": {"type": "int", "default": 70, "min": 60, "max": 80},
        "rsi_oversold": {"type": "int", "default": 30, "min": 20, "max": 40},
        
        "use_moving_averages": {"type": "bool", "default": True},
        "fast_ma_period": {"type": "int", "default": 10, "min": 5, "max": 20},
        "medium_ma_period": {"type": "int", "default": 30, "min": 20, "max": 50},
        "slow_ma_period": {"type": "int", "default": 50, "min": 30, "max": 100},
        
        # Spread selection
        "preferred_spread_types": {"type": "str", "default": "all", 
                                  "enum": ["all", "debit_only", "credit_only", "bullish_only", "bearish_only"]},
        
        "min_risk_reward_ratio": {"type": "float", "default": 0.5, "min": 0.3, "max": 1.0},
        "max_risk_reward_ratio": {"type": "float", "default": 3.0, "min": 1.5, "max": 5.0},
        
        # Position management
        "max_positions": {"type": "int", "default": 5, "min": 1, "max": 10},
        "max_positions_per_direction": {"type": "int", "default": 3, "min": 1, "max": 5},
        
        "profit_target_pct": {"type": "float", "default": 0.5, "min": 0.2, "max": 0.8},
        "max_loss_pct": {"type": "float", "default": 0.5, "min": 0.3, "max": 0.7},
        "max_days_to_hold": {"type": "int", "default": 21, "min": 7, "max": 45},
        
        # Entry/exit logic
        "enter_on_confirmation": {"type": "bool", "default": True},
        "confirmation_periods": {"type": "int", "default": 2, "min": 1, "max": 5},
        "exit_on_confirmation_change": {"type": "bool", "default": True},
        
        # Risk management
        "max_account_risk_pct": {"type": "float", "default": 0.02, "min": 0.01, "max": 0.05},
        "position_size_pct": {"type": "float", "default": 0.02, "min": 0.01, "max": 0.05},
    }
)
class VerticalSpreadEngine(OptionsBaseStrategy):
    """
    A comprehensive vertical spread strategy engine for options trading.
    
    This strategy:
    1. Analyzes market conditions to determine bullish/bearish bias
    2. Selects appropriate vertical spread types based on market conditions
    3. Constructs optimal spreads considering strike selection, risk/reward, etc.
    4. Manages positions with defined entry/exit logic and risk parameters
    5. Adapts to changing market conditions by adjusting spread selection
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize the vertical spread engine.
        
        Args:
            session: Options trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        super().__init__(session, data_pipeline, parameters)
        
        # Initialize specialized components
        spread_analyzer_params = {
            'min_risk_reward_ratio': self.parameters['min_risk_reward_ratio'],
            'max_risk_reward_ratio': self.parameters['max_risk_reward_ratio']
        }
        self.spread_analyzer = SpreadAnalyzer(spread_analyzer_params)
        
        spread_manager_params = {
            'max_positions': self.parameters['max_positions'],
            'max_positions_per_direction': self.parameters['max_positions_per_direction'],
            'profit_target_pct': self.parameters['profit_target_pct'],
            'max_loss_pct': self.parameters['max_loss_pct'],
            'max_days_to_hold': self.parameters['max_days_to_hold']
        }
        self.spread_manager = SpreadManager(spread_manager_params)
        
        # Strategy state
        self.market_bias = "neutral"
        self.market_bias_history = []
        self.confirmation_count = 0
        self.last_analysis_time = None
        self.iv_metrics = {}
        
        logger.info(f"Initialized VerticalSpreadEngine for {session.symbol}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for spread selection and market bias.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        if data.empty or len(data) < 50:
            return {}
        
        indicators = {}
        
        # RSI calculation
        if self.parameters["use_rsi"]:
            period = self.parameters["rsi_period"]
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            indicators["rsi"] = rsi
        
        # Moving averages
        if self.parameters["use_moving_averages"]:
            fast_period = self.parameters["fast_ma_period"]
            medium_period = self.parameters["medium_ma_period"]
            slow_period = self.parameters["slow_ma_period"]
            
            indicators["fast_ma"] = data['close'].rolling(window=fast_period).mean()
            indicators["medium_ma"] = data['close'].rolling(window=medium_period).mean()
            indicators["slow_ma"] = data['close'].rolling(window=slow_period).mean()
            
            # Moving average crossovers
            indicators["fast_above_medium"] = indicators["fast_ma"] > indicators["medium_ma"]
            indicators["medium_above_slow"] = indicators["medium_ma"] > indicators["slow_ma"]
            indicators["all_aligned_bullish"] = (
                indicators["fast_ma"] > indicators["medium_ma"]) & (
                indicators["medium_ma"] > indicators["slow_ma"])
            indicators["all_aligned_bearish"] = (
                indicators["fast_ma"] < indicators["medium_ma"]) & (
                indicators["medium_ma"] < indicators["slow_ma"])
        
        # Price momentum and trend
        if len(data) >= 20:
            indicators["price_20d_change"] = (
                data['close'] / data['close'].shift(20) - 1) * 100
        
        # Volatility
        if len(data) >= 20:
            indicators["20d_volatility"] = data['close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
        
        return indicators
    
    def determine_market_bias(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> str:
        """
        Determine market bias (bullish, bearish, or neutral) based on indicators.
        
        Args:
            data: Market data DataFrame
            indicators: Calculated indicators
            
        Returns:
            Market bias as string
        """
        if data.empty or not indicators:
            return "neutral"
        
        bullish_signals = 0
        bearish_signals = 0
        neutral_threshold = 1  # Minimum signal difference to deviate from neutral
        
        # RSI analysis
        if "rsi" in indicators:
            rsi_value = indicators["rsi"].iloc[-1]
            if rsi_value > self.parameters["rsi_overbought"]:
                bearish_signals += 1
            elif rsi_value < self.parameters["rsi_oversold"]:
                bullish_signals += 1
        
        # Moving average analysis
        if self.parameters["use_moving_averages"]:
            # Aligned moving averages
            if indicators["all_aligned_bullish"].iloc[-1]:
                bullish_signals += 2
            elif indicators["all_aligned_bearish"].iloc[-1]:
                bearish_signals += 2
            # Fast/medium MA crossover
            elif indicators["fast_above_medium"].iloc[-1]:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        # Price momentum
        if "price_20d_change" in indicators:
            momentum = indicators["price_20d_change"].iloc[-1]
            if momentum > 5:  # 5% price increase over last 20 days
                bullish_signals += 1
            elif momentum < -5:  # 5% price decrease over last 20 days
                bearish_signals += 1
        
        # Determine bias
        signal_difference = bullish_signals - bearish_signals
        
        if signal_difference >= neutral_threshold:
            bias = "bullish"
        elif signal_difference <= -neutral_threshold:
            bias = "bearish"
        else:
            bias = "neutral"
        
        # Store for confirmation tracking
        self.market_bias_history.append(bias)
        if len(self.market_bias_history) > 10:
            self.market_bias_history.pop(0)
        
        # Check if we have consistent signals for confirmation
        conf_periods = self.parameters["confirmation_periods"]
        if len(self.market_bias_history) >= conf_periods:
            recent_bias = self.market_bias_history[-conf_periods:]
            if all(b == recent_bias[0] for b in recent_bias):
                self.confirmation_count += 1
                logger.info(f"Market bias confirmed: {bias} (confirmation count: {self.confirmation_count})")
            else:
                self.confirmation_count = 0
        
        return bias
    
    def select_spread_types(self, market_bias: str, iv_percentile: float) -> List[VerticalSpreadType]:
        """
        Select appropriate spread types based on market bias and IV environment.
        
        Args:
            market_bias: Current market bias
            iv_percentile: Current IV percentile
            
        Returns:
            List of vertical spread types to consider
        """
        preferred = self.parameters["preferred_spread_types"]
        
        # Start with all spread types
        all_spread_types = list(VerticalSpreadType)
        
        # Filter based on preferences
        if preferred == "debit_only":
            spread_types = [t for t in all_spread_types if VerticalSpreadType.is_debit(t)]
        elif preferred == "credit_only":
            spread_types = [t for t in all_spread_types if VerticalSpreadType.is_credit(t)]
        elif preferred == "bullish_only":
            spread_types = [t for t in all_spread_types if VerticalSpreadType.is_bullish(t)]
        elif preferred == "bearish_only":
            spread_types = [t for t in all_spread_types if VerticalSpreadType.is_bearish(t)]
        else:
            spread_types = all_spread_types
        
        # Filter based on market bias
        if market_bias == "bullish":
            spread_types = [t for t in spread_types if VerticalSpreadType.is_bullish(t)]
        elif market_bias == "bearish":
            spread_types = [t for t in spread_types if VerticalSpreadType.is_bearish(t)]
        
        # If no spreads match the criteria, use neutral approach (include all)
        if not spread_types:
            spread_types = all_spread_types
        
        # Prioritize spread types based on IV environment
        if iv_percentile > 70:  # High IV environment - favor credit spreads
            spread_types.sort(key=lambda t: 0 if VerticalSpreadType.is_credit(t) else 1)
        elif iv_percentile < 30:  # Low IV environment - favor debit spreads
            spread_types.sort(key=lambda t: 0 if VerticalSpreadType.is_debit(t) else 1)
        
        return spread_types
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate vertical spread trading signals based on market conditions.
        
        Args:
            data: Market data DataFrame
            indicators: Calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            "enter_spread": False,
            "exit_positions": [],
            "spread_type": None,
            "market_bias": "neutral",
            "signal_strength": 0.0
        }
        
        if data.empty or not indicators or self.session.option_chain is None:
            return signals
        
        # Get current underlying price
        current_price = data['close'].iloc[-1]
        
        # Get current date
        current_date = data.index[-1].date() if isinstance(data.index[-1], pd.Timestamp) else datetime.now().date()
        
        # Determine market bias
        market_bias = self.determine_market_bias(data, indicators)
        signals["market_bias"] = market_bias
        self.market_bias = market_bias
        
        # Calculate IV metrics
        if self.session.current_iv is not None and self.session.symbol in self.iv_history:
            self.iv_metrics = self.calculate_implied_volatility_metrics(self.iv_history[self.session.symbol])
            iv_percentile = self.iv_metrics.get('iv_percentile', 50)
        else:
            iv_percentile = 50  # Default to middle if no data
        
        # Update positions with latest market data
        self.spread_manager.update_positions(self.session.option_chain, current_price, current_date)
        
        # Entry logic
        enter_confirmed = (self.parameters["enter_on_confirmation"] and 
                         self.confirmation_count >= self.parameters["confirmation_periods"])
        
        if (enter_confirmed or not self.parameters["enter_on_confirmation"]) and market_bias != "neutral":
            # Select spread types based on market conditions
            spread_types = self.select_spread_types(market_bias, iv_percentile)
            
            # Check if we can open new positions
            for spread_type in spread_types:
                if self.spread_manager.can_open_position(spread_type):
                    # Try to construct a valid spread
                    spread = self.spread_analyzer.construct_vertical_spread(
                        spread_type, 
                        self.session.option_chain,
                        current_price,
                        market_bias,
                        iv_percentile
                    )
                    
                    if spread:
                        signals["enter_spread"] = True
                        signals["spread_type"] = spread_type
                        signals["spread"] = spread
                        signals["signal_strength"] = 0.7 + (self.confirmation_count * 0.05)  # Increase with confirmations
                        break
        
        # Exit logic - check for bias reversals
        if self.parameters["exit_on_confirmation_change"] and len(self.market_bias_history) >= 2:
            previous_bias = self.market_bias_history[-2]
            current_bias = self.market_bias_history[-1]
            
            if previous_bias != current_bias and previous_bias != "neutral":
                # Bias has changed, check for positions in the opposite direction
                for position_id, position in self.spread_manager.active_positions.items():
                    spread_type = position.spread.spread_type
                    is_bullish = VerticalSpreadType.is_bullish(spread_type)
                    
                    # Exit bullish positions when bias turns bearish
                    if is_bullish and current_bias == "bearish":
                        signals["exit_positions"].append(position_id)
                    
                    # Exit bearish positions when bias turns bullish
                    elif not is_bullish and current_bias == "bullish":
                        signals["exit_positions"].append(position_id)
        
        return signals
    
    def _execute_signals(self):
        """Execute trading signals generated by the strategy."""
        if not self.signals:
            return
        
        # Handle entry signals
        if self.signals.get("enter_spread", False) and "spread" in self.signals:
            spread = self.signals["spread"]
            underlying_price = self.session.current_price or self.market_data['close'].iloc[-1]
            
            # Open the position
            position_id = self.spread_manager.open_position(spread, underlying_price)
            
            if position_id:
                logger.info(f"Opened {spread.spread_type.value} position {position_id}")
        
        # Handle exit signals
        for position_id in self.signals.get("exit_positions", []):
            if self.spread_manager.close_position(position_id, "signal_reversal"):
                logger.info(f"Closed position {position_id} due to market bias change")
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get the current state of the strategy."""
        return {
            "market_bias": self.market_bias,
            "confirmation_count": self.confirmation_count,
            "iv_metrics": self.iv_metrics,
            "positions": self.spread_manager.get_position_summary(),
            "last_analysis_time": self.last_analysis_time
        }
