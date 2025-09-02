#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta-Memory Strategy

This strategy extends the ML Strategy with meta-memory for enhanced signal context.
It tracks historical performance in different market conditions to adjust signal
confidence and position sizing adaptively.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

from trading_bot.strategies.ml_strategy import MLStrategy
from trading_bot.strategies.strategy_template import Signal, SignalType, TimeFrame
from trading_bot.utils.meta_memory import MetaMemory
from trading_bot.utils.feature_engineering import FeatureEngineering

logger = logging.getLogger(__name__)

class MetaMemoryStrategy(MLStrategy):
    """
    Enhanced ML Strategy with meta-memory for signal context awareness.
    
    Key features:
    1. Records past signals and their outcomes
    2. Adjusts confidence based on historical performance in similar contexts
    3. Learns to avoid false signals
    4. Provides enhanced signal context for better decision-making
    5. Persists memory across strategy runs for continuous learning
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        memory_path: str = None,
        memory_config: Dict[str, Any] = None
    ):
        """
        Initialize the meta-memory strategy.
        
        Args:
            config: Strategy configuration parameters
            memory_path: Path to save/load memory state (default to trading_data/meta_memory/{strategy_name}.pkl)
            memory_config: Meta-memory specific configuration
        """
        super().__init__(config)
        
        # Set default memory path if not provided
        if memory_path is None:
            strategy_name = self.config.get('strategy_name', 'meta_memory')
            memory_dir = os.path.join('trading_data', 'meta_memory')
            os.makedirs(memory_dir, exist_ok=True)
            memory_path = os.path.join(memory_dir, f"{strategy_name}.pkl")
        
        # Configure memory
        self.memory_config = memory_config or {}
        self.memory_config.setdefault('max_memory_size', 1000)
        
        # Initialize meta-memory system
        self.meta_memory = MetaMemory(
            config=self.memory_config,
            memory_path=memory_path,
            max_memory_size=self.memory_config.get('max_memory_size', 1000)
        )
        
        # Track active signals
        self.active_signals = {}
        
        # Configure memory save frequency (default: every 10 signals)
        self.save_frequency = self.memory_config.get('save_frequency', 10)
        self.signal_counter = 0
        
        logger.info(f"Meta-Memory Strategy initialized with memory path: {memory_path}")
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: TimeFrame
    ) -> Signal:
        """
        Generate trading signal with meta-memory enhancement.
        
        Args:
            data: Historical market data
            symbol: Trading symbol
            timeframe: Signal timeframe
            
        Returns:
            Enhanced trading signal
        """
        # Get base signal from ML strategy
        signal = super().generate_signal(data, symbol, timeframe)
        
        # Extract market context
        context = self._extract_market_context(data, symbol)
        
        # Store the original signal in memory
        self.meta_memory.store_signal(symbol, signal, context)
        
        # Only enhance if we have enough historical signals
        symbol_memory = self.meta_memory.signal_memory.get(symbol, None)
        if symbol_memory and len(symbol_memory) >= 5:
            enhanced_signal = self.meta_memory.enhance_signal(symbol, signal, context)
            
            # Log enhancement if significant adjustment was made
            if signal.confidence != enhanced_signal.confidence:
                adj_pct = ((enhanced_signal.confidence / signal.confidence) - 1.0) * 100
                logger.info(f"Signal confidence adjusted by {adj_pct:.1f}% based on meta-memory")
            
            signal = enhanced_signal
        
        # Track active signals
        if signal.signal_type != SignalType.NEUTRAL:
            self.active_signals[symbol] = {
                'signal': signal,
                'entry_time': signal.timestamp,
                'entry_price': signal.price,
                'last_update': datetime.now()
            }
        
        # Auto-save memory periodically
        self.signal_counter += 1
        if self.signal_counter >= self.save_frequency:
            self.meta_memory.save_memory()
            self.signal_counter = 0
        
        return signal
    
    def _extract_market_context(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Extract market context information from historical data.
        
        Args:
            data: Historical market data
            symbol: Trading symbol
            
        Returns:
            Dict containing market context information
        """
        context = {
            'timestamp': datetime.now(),
            'symbol': symbol
        }
        
        # Ensure we have enough data
        if len(data) < 20:
            return context
        
        # Calculate volatility (20-period ATR normalized by price)
        if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values[:-1]  # Previous close
            
            tr1 = np.abs(high[1:] - low[1:])
            tr2 = np.abs(high[1:] - close)
            tr3 = np.abs(low[1:] - close)
            
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = np.mean(tr[-20:])
            
            # Normalize by current price
            context['volatility'] = atr / data['close'].iloc[-1]
        
        # Calculate trend strength
        if 'close' in data.columns:
            prices = data['close'].values
            
            # Use simple linear regression slope as trend measure
            x = np.arange(min(20, len(prices)))
            y = prices[-min(20, len(prices)):]
            
            if len(x) == len(y) and len(x) > 1:
                slope, _ = np.polyfit(x, y, 1)
                # Normalize by price level and period length
                context['trend_strength'] = slope * len(x) / y[-1]
            
            # Add price momentum
            if len(prices) >= 10:
                momentum_10 = (prices[-1] / prices[-10]) - 1.0
                context['momentum_10'] = momentum_10
        
        # Calculate volume ratio if volume available
        if 'volume' in data.columns:
            recent_vol = data['volume'].iloc[-5:].mean()
            longer_vol = data['volume'].iloc[-20:].mean()
            
            if longer_vol > 0:
                context['volume_ratio'] = recent_vol / longer_vol
        
        # Determine market regime
        context['market_regime'] = self._determine_market_regime(data)
        
        return context
    
    def _determine_market_regime(self, data: pd.DataFrame) -> str:
        """
        Determine the current market regime based on price action.
        
        Args:
            data: Historical market data
            
        Returns:
            Market regime identifier
        """
        # Default regime
        regime = "unknown"
        
        # Ensure we have enough data
        if len(data) < 50 or 'close' not in data.columns:
            return regime
        
        close = data['close'].values
        
        # Calculate short and long moving averages
        ma_short = np.mean(close[-20:])
        ma_long = np.mean(close[-50:])
        
        # Calculate recent volatility
        returns = np.diff(close[-30:]) / close[-31:-1]
        recent_vol = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Median historical volatility for comparison
        if len(returns) >= 20:
            hist_vol_windows = [np.std(returns[i:i+20]) * np.sqrt(252) for i in range(len(returns)-20)]
            median_vol = np.median(hist_vol_windows) if hist_vol_windows else recent_vol
        else:
            median_vol = recent_vol
        
        # Determine trend direction
        trend_up = ma_short > ma_long * 1.01
        trend_down = ma_short < ma_long * 0.99
        
        # Determine volatility regime
        high_vol = recent_vol > median_vol * 1.5
        low_vol = recent_vol < median_vol * 0.5
        
        # Classify regime
        if trend_up:
            if high_vol:
                regime = "volatile_uptrend"
            elif low_vol:
                regime = "steady_uptrend"
            else:
                regime = "uptrend"
        elif trend_down:
            if high_vol:
                regime = "volatile_downtrend"
            elif low_vol:
                regime = "steady_downtrend"
            else:
                regime = "downtrend"
        else:
            if high_vol:
                regime = "volatile_range"
            elif low_vol:
                regime = "tight_range"
            else:
                regime = "range"
        
        return regime
    
    def update_active_signals(
        self,
        current_prices: Dict[str, float]
    ) -> None:
        """
        Update active signals with current price information.
        
        Args:
            current_prices: Dictionary of current prices by symbol
        """
        for symbol, signal_info in list(self.active_signals.items()):
            if symbol in current_prices:
                current_price = current_prices[symbol]
                signal = signal_info['signal']
                entry_price = signal_info['entry_price']
                
                # Calculate current P&L
                if signal.signal_type == SignalType.BUY:
                    pnl_pct = (current_price / entry_price) - 1.0
                elif signal.signal_type == SignalType.SELL:
                    pnl_pct = 1.0 - (current_price / entry_price)
                else:
                    pnl_pct = 0.0
                
                # Update signal info
                signal_info['current_price'] = current_price
                signal_info['current_pnl'] = pnl_pct
                signal_info['last_update'] = datetime.now()
                
                # Check for stop loss or take profit levels
                if signal.stop_loss is not None:
                    if (signal.signal_type == SignalType.BUY and current_price <= signal.stop_loss) or \
                       (signal.signal_type == SignalType.SELL and current_price >= signal.stop_loss):
                        # Stop loss hit
                        self._process_signal_outcome(
                            symbol, 
                            signal_info, 
                            False, 
                            "stop_loss", 
                            current_price
                        )
                
                if signal.take_profit is not None:
                    if (signal.signal_type == SignalType.BUY and current_price >= signal.take_profit) or \
                       (signal.signal_type == SignalType.SELL and current_price <= signal.take_profit):
                        # Take profit hit
                        self._process_signal_outcome(
                            symbol, 
                            signal_info, 
                            True, 
                            "take_profit", 
                            current_price
                        )
    
    def _process_signal_outcome(
        self,
        symbol: str,
        signal_info: Dict[str, Any],
        success: bool,
        exit_reason: str,
        exit_price: float
    ) -> None:
        """
        Process a signal outcome.
        
        Args:
            symbol: Trading symbol
            signal_info: Active signal information
            success: Whether the signal was successful
            exit_reason: Reason for signal exit
            exit_price: Exit price
        """
        entry_price = signal_info['entry_price']
        signal = signal_info['signal']
        
        # Calculate P&L
        if signal.signal_type == SignalType.BUY:
            pnl_pct = (exit_price / entry_price) - 1.0
        elif signal.signal_type == SignalType.SELL:
            pnl_pct = 1.0 - (exit_price / entry_price)
        else:
            pnl_pct = 0.0
        
        # Prepare outcome data
        outcome = {
            'exit_time': datetime.now(),
            'exit_price': exit_price,
            'entry_price': entry_price,
            'pnl_pct': pnl_pct,
            'success': success,
            'exit_reason': exit_reason,
            'holding_period': (datetime.now() - signal_info['entry_time']).total_seconds() / 3600  # in hours
        }
        
        # Store outcome in meta-memory
        self.meta_memory.store_outcome(symbol, signal_info['entry_time'], outcome)
        
        # Remove from active signals
        if symbol in self.active_signals:
            del self.active_signals[symbol]
        
        # Log outcome
        logger.info(
            f"Signal outcome for {symbol}: {exit_reason}, PnL: {pnl_pct:.2%}, "
            f"Success: {success}, Holding period: {outcome['holding_period']:.1f} hours"
        )
    
    def close_open_positions(
        self,
        current_prices: Dict[str, float],
        reason: str = "manual_close"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Close all open positions and record outcomes.
        
        Args:
            current_prices: Dictionary of current prices by symbol
            reason: Reason for closing positions
            
        Returns:
            Dictionary of outcomes by symbol
        """
        outcomes = {}
        
        for symbol, signal_info in list(self.active_signals.items()):
            if symbol in current_prices:
                current_price = current_prices[symbol]
                signal = signal_info['signal']
                entry_price = signal_info['entry_price']
                
                # Calculate P&L
                if signal.signal_type == SignalType.BUY:
                    pnl_pct = (current_price / entry_price) - 1.0
                elif signal.signal_type == SignalType.SELL:
                    pnl_pct = 1.0 - (current_price / entry_price)
                else:
                    pnl_pct = 0.0
                
                # Determine success (positive P&L)
                success = pnl_pct > 0
                
                # Process outcome
                self._process_signal_outcome(
                    symbol,
                    signal_info,
                    success,
                    reason,
                    current_price
                )
                
                # Store outcome
                outcomes[symbol] = {
                    'pnl_pct': pnl_pct,
                    'success': success,
                    'exit_price': current_price,
                    'entry_price': entry_price,
                    'signal_type': signal.signal_type.value
                }
        
        # Save memory after closing positions
        self.meta_memory.save_memory()
        
        return outcomes
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the meta-memory system.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.meta_memory.get_statistics()
        stats['active_signals'] = len(self.active_signals)
        
        # Add active signal details
        stats['active_signals_details'] = {
            symbol: {
                'signal_type': info['signal'].signal_type.value,
                'entry_time': info['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': info['entry_price'],
                'current_pnl': info.get('current_pnl', None),
                'confidence': info['signal'].confidence
            }
            for symbol, info in self.active_signals.items()
        }
        
        return stats 