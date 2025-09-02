#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta-Memory Signal Context

This module implements meta-memory for trading signals, providing context
awareness and memory of past signals, market conditions, and performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, DefaultDict
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import os
import joblib

from trading_bot.strategies.strategy_template import Signal, SignalType, TimeFrame

logger = logging.getLogger(__name__)

class MetaMemory:
    """
    Meta-Memory system that maintains historical records of signals, contexts and outcomes
    to enhance trading decision quality through contextual awareness.
    
    Key features:
    1. Signal adjustment based on past performance in similar market conditions
    2. Detection of false signal patterns for improved filtering
    3. Adaptive confidence scoring based on historical effectiveness
    4. Persistence of memory across trading sessions
    5. Symbol-specific memory contextualization
    """
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        memory_path: str = None,
        max_memory_size: int = 1000
    ):
        """
        Initialize the meta-memory system.
        
        Args:
            config: Configuration parameters
            memory_path: Path to save/load memory state
            max_memory_size: Maximum number of signals to keep in memory per symbol
        """
        self.config = config or {}
        self.memory_path = memory_path
        self.max_memory_size = max_memory_size
        
        # Initialize memory structures
        self.signal_memory: DefaultDict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.max_memory_size)
        )
        self.context_memory: DefaultDict[str, Dict] = defaultdict(dict)
        self.performance_memory: DefaultDict[str, Dict] = defaultdict(
            lambda: {
                'buy': {'count': 0, 'success': 0, 'pnl_sum': 0.0, 'confidence_sum': 0.0},
                'sell': {'count': 0, 'success': 0, 'pnl_sum': 0.0, 'confidence_sum': 0.0},
                'by_regime': defaultdict(
                    lambda: {'count': 0, 'success': 0, 'pnl_sum': 0.0}
                ),
                'total_signals': 0,
                'successful_signals': 0,
                'avg_pnl': 0.0,
                'last_update': None
            }
        )
        
        # Configure enhancement parameters
        self.context_weight = self.config.get('context_weight', 0.4)
        self.recency_weight = self.config.get('recency_weight', 0.3)
        self.pattern_weight = self.config.get('pattern_weight', 0.3)
        
        # Load existing memory if available
        if self.memory_path and os.path.exists(self.memory_path):
            self.load_memory()
    
    def store_signal(
        self,
        symbol: str,
        signal: Signal,
        context: Dict[str, Any]
    ) -> None:
        """
        Store a signal with its context in memory.
        
        Args:
            symbol: Trading symbol
            signal: The trading signal
            context: Market context information
        """
        # Store signal in memory
        signal_data = {
            'timestamp': signal.timestamp,
            'signal_type': signal.signal_type.value,
            'confidence': signal.confidence,
            'price': signal.price,
            'context': context.copy(),
            'metadata': signal.metadata.copy() if signal.metadata else {},
            'outcome': None  # Will be updated when outcome is known
        }
        
        self.signal_memory[symbol].append(signal_data)
        
        # Update context memory statistics
        regime = context.get('market_regime', 'unknown')
        if regime not in self.context_memory[symbol]:
            self.context_memory[symbol][regime] = {
                'count': 0,
                'buy_count': 0,
                'sell_count': 0,
                'success_rate': 0.0,
                'avg_pnl': 0.0,
                'volatility_avg': 0.0,
                'signals': []
            }
        
        # Update regime counts
        self.context_memory[symbol][regime]['count'] += 1
        if signal.signal_type == SignalType.BUY:
            self.context_memory[symbol][regime]['buy_count'] += 1
        elif signal.signal_type == SignalType.SELL:
            self.context_memory[symbol][regime]['sell_count'] += 1
        
        # Update volatility average
        if 'volatility' in context:
            old_avg = self.context_memory[symbol][regime]['volatility_avg']
            old_count = self.context_memory[symbol][regime]['count'] - 1
            new_vol = context['volatility']
            
            if old_count == 0:
                self.context_memory[symbol][regime]['volatility_avg'] = new_vol
            else:
                self.context_memory[symbol][regime]['volatility_avg'] = \
                    (old_avg * old_count + new_vol) / (old_count + 1)
        
        # Keep reference to recent signals in this regime (limited to last 20)
        signal_refs = self.context_memory[symbol][regime]['signals']
        signal_refs.append(len(self.signal_memory[symbol]) - 1)
        if len(signal_refs) > 20:
            signal_refs.pop(0)

    def store_outcome(
        self,
        symbol: str,
        entry_time: datetime,
        outcome: Dict[str, Any]
    ) -> bool:
        """
        Store the outcome of a signal and update effectiveness metrics.
        
        Args:
            symbol: Trading symbol
            entry_time: Entry timestamp of the signal
            outcome: Outcome information
            
        Returns:
            True if outcome was stored, False if signal not found
        """
        if symbol not in self.signal_memory or not self.signal_memory[symbol]:
            return False
        
        # Find the signal by timestamp
        signal_found = False
        for i, signal_data in enumerate(self.signal_memory[symbol]):
            if abs((signal_data['timestamp'] - entry_time).total_seconds()) < 10:  # Within 10 seconds
                # Update the outcome
                self.signal_memory[symbol][i]['outcome'] = outcome
                signal_found = True
                
                # Get signal info
                signal_type = signal_data['signal_type']
                confidence = signal_data['confidence']
                context = signal_data['context']
                regime = context.get('market_regime', 'unknown')
                success = outcome['success']
                pnl_pct = outcome['pnl_pct']
                
                # Update performance memory
                perf = self.performance_memory[symbol]
                perf['total_signals'] += 1
                perf['last_update'] = datetime.now()
                
                if success:
                    perf['successful_signals'] += 1
                
                # Update PnL tracking
                old_avg_pnl = perf['avg_pnl'] 
                old_count = perf['total_signals'] - 1
                
                if old_count == 0:
                    perf['avg_pnl'] = pnl_pct
                else:
                    perf['avg_pnl'] = (old_avg_pnl * old_count + pnl_pct) / perf['total_signals']
                
                # Update signal type specific stats
                signal_type_key = signal_type.lower() if isinstance(signal_type, str) else 'neutral'
                if signal_type_key in ('buy', 'sell'):
                    signal_stats = perf[signal_type_key]
                    signal_stats['count'] += 1
                    signal_stats['pnl_sum'] += pnl_pct
                    signal_stats['confidence_sum'] += confidence
                    
                    if success:
                        signal_stats['success'] += 1
                
                # Update regime-specific performance
                regime_stats = perf['by_regime'][regime]
                regime_stats['count'] += 1
                regime_stats['pnl_sum'] += pnl_pct
                
                if success:
                    regime_stats['success'] += 1
                
                # Update context memory statistics
                if regime in self.context_memory[symbol]:
                    context_stats = self.context_memory[symbol][regime]
                    old_success = context_stats['success_rate'] * context_stats.get('outcomes_count', 0)
                    old_pnl = context_stats['avg_pnl'] * context_stats.get('outcomes_count', 0)
                    
                    # Increment outcomes count or initialize
                    if 'outcomes_count' in context_stats:
                        context_stats['outcomes_count'] += 1
                    else:
                        context_stats['outcomes_count'] = 1
                    
                    # Update success rate and avg PnL
                    outcomes_count = context_stats['outcomes_count']
                    context_stats['success_rate'] = (old_success + (1 if success else 0)) / outcomes_count
                    context_stats['avg_pnl'] = (old_pnl + pnl_pct) / outcomes_count
                
                break
        
        return signal_found
    
    def enhance_signal(
        self,
        symbol: str,
        signal: Signal,
        context: Dict[str, Any]
    ) -> Signal:
        """
        Enhance a trading signal using meta-memory context awareness.
        
        Args:
            symbol: Trading symbol
            signal: The original trading signal
            context: Current market context
            
        Returns:
            Enhanced signal with adjustments
        """
        if signal.signal_type == SignalType.NEUTRAL:
            return signal  # Don't enhance neutral signals
        
        # Create a copy of the signal to enhance
        enhanced = Signal(
            timestamp=signal.timestamp,
            signal_type=signal.signal_type,
            symbol=signal.symbol,
            price=signal.price,
            confidence=signal.confidence,
            timeframe=signal.timeframe,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            metadata=signal.metadata.copy() if signal.metadata else {}
        )
        
        # Apply enhancements if we have sufficient history
        if symbol in self.signal_memory and len(self.signal_memory[symbol]) >= 3:
            # Calculate adjustments based on different factors
            context_adjustment = self._calculate_context_adjustment(symbol, signal, context)
            recency_adjustment = self._calculate_recency_adjustment(symbol, signal)
            pattern_adjustment = self._calculate_pattern_adjustment(symbol, signal, context)
            
            # Combine adjustments with weights
            total_adjustment = (
                self.context_weight * context_adjustment +
                self.recency_weight * recency_adjustment +
                self.pattern_weight * pattern_adjustment
            )
            
            # Apply adjustment to confidence (limited to ±30%)
            total_adjustment = max(-0.3, min(0.3, total_adjustment))
            original_confidence = enhanced.confidence
            enhanced.confidence = max(0.01, min(0.99, original_confidence * (1 + total_adjustment)))
            
            # Store enhancement data in metadata
            if enhanced.metadata is None:
                enhanced.metadata = {}
            
            enhanced.metadata['meta_memory'] = {
                'original_confidence': original_confidence,
                'context_adjustment': context_adjustment,
                'recency_adjustment': recency_adjustment,
                'pattern_adjustment': pattern_adjustment,
                'total_adjustment': total_adjustment,
                'context': {k: v for k, v in context.items() if k != 'timestamp'}
            }
            
            # Adjust stop loss and take profit if significant adjustment
            if abs(total_adjustment) > 0.1 and enhanced.stop_loss is not None:
                # For positive adjustments, tighten stops
                # For negative adjustments, loosen stops
                sl_adjustment = 1.0 - (total_adjustment * 0.5)  # Scale adjustment for stops
                
                if signal.signal_type == SignalType.BUY:
                    # For buy signals, stop loss is below entry
                    sl_distance = signal.price - enhanced.stop_loss
                    enhanced.stop_loss = signal.price - (sl_distance * sl_adjustment)
                    
                    # Adjust take profit if present
                    if enhanced.take_profit is not None:
                        tp_distance = enhanced.take_profit - signal.price
                        enhanced.take_profit = signal.price + (tp_distance * (1.0 + (total_adjustment * 0.3)))
                
                elif signal.signal_type == SignalType.SELL:
                    # For sell signals, stop loss is above entry
                    sl_distance = enhanced.stop_loss - signal.price
                    enhanced.stop_loss = signal.price + (sl_distance * sl_adjustment)
                    
                    # Adjust take profit if present
                    if enhanced.take_profit is not None:
                        tp_distance = signal.price - enhanced.take_profit
                        enhanced.take_profit = signal.price - (tp_distance * (1.0 + (total_adjustment * 0.3)))
        
        return enhanced
    
    def _calculate_context_adjustment(
        self,
        symbol: str,
        signal: Signal,
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence adjustment based on performance in similar market contexts.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            context: Current market context
            
        Returns:
            Adjustment factor (-1.0 to 1.0)
        """
        # Default neutral adjustment
        adjustment = 0.0
        
        # Get current regime
        regime = context.get('market_regime', 'unknown')
        
        # Check if we have data for this regime
        if regime in self.context_memory[symbol]:
            regime_data = self.context_memory[symbol][regime]
            
            # Only adjust if we have outcomes in this regime
            if regime_data.get('outcomes_count', 0) >= 3:
                # Base adjustment on success rate relative to baseline (0.5)
                success_rate = regime_data['success_rate']
                pnl_score = min(1.0, max(-1.0, regime_data['avg_pnl'] * 10))  # Scale PnL to -1 to 1
                
                # Combine success rate and PnL for adjustment
                baseline = 0.5  # Expected success rate by chance
                adjustment = ((success_rate - baseline) * 0.7) + (pnl_score * 0.3)
                
                # Apply signal type specific adjustment
                signal_type_key = signal.signal_type.value.lower()
                if signal_type_key in ('buy', 'sell'):
                    type_count = regime_data.get(f'{signal_type_key}_count', 0)
                    
                    # If this type of signal is rare in this regime, reduce confidence
                    if type_count <= 2 and regime_data['count'] >= 10:
                        adjustment -= 0.1
        
        # Check volatility context
        if 'volatility' in context and regime in self.context_memory[symbol]:
            current_vol = context['volatility']
            avg_vol = self.context_memory[symbol][regime].get('volatility_avg', current_vol)
            
            # If current volatility is much higher than average, reduce confidence
            if current_vol > avg_vol * 1.5:
                adjustment -= 0.1
            # If current volatility is much lower than average, increase confidence
            elif current_vol < avg_vol * 0.5:
                adjustment += 0.05
        
        return adjustment
    
    def _calculate_recency_adjustment(
        self,
        symbol: str,
        signal: Signal
    ) -> float:
        """
        Calculate confidence adjustment based on recent performance.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            
        Returns:
            Adjustment factor (-1.0 to 1.0)
        """
        # Default neutral adjustment
        adjustment = 0.0
        
        # Get recent signals with outcomes
        recent_signals = deque(maxlen=10)
        for s in reversed(self.signal_memory[symbol]):
            if s['outcome'] is not None:
                recent_signals.appendleft(s)
                if len(recent_signals) >= 10:
                    break
        
        if not recent_signals:
            return adjustment
        
        # Calculate success rate and PnL for recent signals
        success_count = sum(1 for s in recent_signals if s['outcome']['success'])
        success_rate = success_count / len(recent_signals)
        
        # Calculate recent PnL
        recent_pnl = sum(s['outcome']['pnl_pct'] for s in recent_signals) / len(recent_signals)
        
        # Calculate streak (consecutive wins or losses)
        streak = 0
        last_success = None
        for s in recent_signals:
            current_success = s['outcome']['success']
            
            if last_success is None:
                last_success = current_success
                streak = 1
            elif current_success == last_success:
                streak += 1
            else:
                break
        
        # Apply adjustments based on recent performance
        baseline = 0.5
        adjustment = (success_rate - baseline) * 0.5
        
        # Add PnL contribution
        pnl_factor = min(1.0, max(-1.0, recent_pnl * 10))
        adjustment += pnl_factor * 0.3
        
        # Add streak contribution (reduce confidence on losing streaks, increase on winning)
        if streak >= 3:
            if last_success:
                adjustment += min(0.1, streak * 0.02)  # Winning streak
            else:
                adjustment -= min(0.2, streak * 0.04)  # Losing streak
        
        return adjustment
    
    def _calculate_pattern_adjustment(
        self,
        symbol: str,
        signal: Signal,
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence adjustment based on pattern recognition of false signals.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            context: Current market context
            
        Returns:
            Adjustment factor (-1.0 to 1.0)
        """
        # Default neutral adjustment
        adjustment = 0.0
        
        # Get signal characteristics
        signal_type = signal.signal_type.value.lower()
        if signal_type not in ('buy', 'sell'):
            return adjustment
        
        # Look for similar contexts that led to false signals
        similar_contexts = []
        
        for s in self.signal_memory[symbol]:
            # Skip signals without outcomes
            if s['outcome'] is None:
                continue
            
            # Only look at same signal type
            if s['signal_type'].lower() != signal_type:
                continue
            
            # Check context similarity
            if self._contexts_are_similar(s['context'], context):
                similar_contexts.append(s)
        
        # If we have similar contexts, analyze outcomes
        if similar_contexts:
            # Calculate success rate
            success_count = sum(1 for s in similar_contexts if s['outcome']['success'])
            success_rate = success_count / len(similar_contexts)
            
            # Calculate average PnL
            avg_pnl = sum(s['outcome']['pnl_pct'] for s in similar_contexts) / len(similar_contexts)
            
            # Special pattern: high confidence signals that failed
            high_conf_failures = sum(1 for s in similar_contexts 
                                     if s['confidence'] > 0.7 and not s['outcome']['success'])
            
            if high_conf_failures >= 2 and signal.confidence > 0.7:
                adjustment -= 0.15
            
            # Adjust based on success rate of similar contexts
            baseline = 0.5  # Expected success rate by chance
            
            # Only make significant adjustments with sufficient data points
            if len(similar_contexts) >= 3:
                adjustment += (success_rate - baseline) * 0.6
                
                # Add PnL contribution to adjustment
                pnl_factor = min(1.0, max(-1.0, avg_pnl * 10))
                adjustment += pnl_factor * 0.2
        
        return adjustment
    
    def _contexts_are_similar(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any]
    ) -> bool:
        """
        Determine if two market contexts are similar.
        
        Args:
            context1: First context
            context2: Second context
            
        Returns:
            True if contexts are similar, False otherwise
        """
        # Must have same regime
        if context1.get('market_regime') != context2.get('market_regime'):
            return False
        
        # Check volatility similarity if available
        if 'volatility' in context1 and 'volatility' in context2:
            vol1 = context1['volatility']
            vol2 = context2['volatility']
            
            # Volatility should be within 50% of each other
            if not (0.5 <= vol1 / vol2 <= 2.0):
                return False
        
        # Check trend similarity if available
        if 'trend_strength' in context1 and 'trend_strength' in context2:
            trend1 = context1['trend_strength']
            trend2 = context2['trend_strength']
            
            # Trend should have same sign and similar magnitude
            if np.sign(trend1) != np.sign(trend2):
                return False
            
            # Trend magnitude should be within 100% of each other
            if abs(trend1) > 0.01 and abs(trend2) > 0.01:
                if not (0.25 <= abs(trend1) / abs(trend2) <= 4.0):
                    return False
        
        # Volume similarity if available
        if 'volume_ratio' in context1 and 'volume_ratio' in context2:
            vol_ratio1 = context1['volume_ratio']
            vol_ratio2 = context2['volume_ratio']
            
            # Volume ratios should be similar
            if not (0.5 <= vol_ratio1 / vol_ratio2 <= 2.0):
                return False
        
        # If we passed all the similarity checks, contexts are similar
        return True
    
    def save_memory(self) -> bool:
        """
        Save the memory state to disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.memory_path:
            logger.warning("Cannot save memory - no memory path specified")
            return False
        
        try:
            memory_dir = os.path.dirname(self.memory_path)
            if memory_dir and not os.path.exists(memory_dir):
                os.makedirs(memory_dir, exist_ok=True)
            
            # Save memory structures
            memory_data = {
                'signal_memory': {k: list(v) for k, v in self.signal_memory.items()},
                'context_memory': dict(self.context_memory),
                'performance_memory': dict(self.performance_memory),
                'config': self.config,
                'max_memory_size': self.max_memory_size,
                'saved_at': datetime.now()
            }
            
            joblib.dump(memory_data, self.memory_path)
            logger.info(f"Meta-memory saved to {self.memory_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save memory: {str(e)}")
            return False
    
    def load_memory(self) -> bool:
        """
        Load the memory state from disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.memory_path or not os.path.exists(self.memory_path):
            logger.warning(f"Memory file not found: {self.memory_path}")
            return False
        
        try:
            memory_data = joblib.load(self.memory_path)
            
            # Restore signal memory (convert lists back to deques)
            for symbol, signals in memory_data['signal_memory'].items():
                self.signal_memory[symbol] = deque(signals, maxlen=self.max_memory_size)
            
            # Restore other memory structures
            self.context_memory = defaultdict(dict, memory_data['context_memory'])
            self.performance_memory = defaultdict(
                lambda: {
                    'buy': {'count': 0, 'success': 0, 'pnl_sum': 0.0, 'confidence_sum': 0.0},
                    'sell': {'count': 0, 'success': 0, 'pnl_sum': 0.0, 'confidence_sum': 0.0},
                    'by_regime': defaultdict(
                        lambda: {'count': 0, 'success': 0, 'pnl_sum': 0.0}
                    ),
                    'total_signals': 0,
                    'successful_signals': 0,
                    'avg_pnl': 0.0,
                    'last_update': None
                }, 
                memory_data['performance_memory']
            )
            
            # Update config if needed
            if 'config' in memory_data:
                self.config.update(memory_data['config'])
            
            # Update max memory size if different
            if 'max_memory_size' in memory_data and memory_data['max_memory_size'] != self.max_memory_size:
                self.max_memory_size = memory_data['max_memory_size']
                # Update maxlen of existing deques
                for symbol in self.signal_memory:
                    old_deque = self.signal_memory[symbol]
                    self.signal_memory[symbol] = deque(old_deque, maxlen=self.max_memory_size)
            
            logger.info(f"Meta-memory loaded from {self.memory_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load memory: {str(e)}")
            return False
    
    def reset_memory(self, symbol: Optional[str] = None) -> None:
        """
        Reset the memory for a specific symbol or all symbols.
        
        Args:
            symbol: Symbol to reset memory for, or None for all symbols
        """
        if symbol:
            if symbol in self.signal_memory:
                self.signal_memory[symbol].clear()
            if symbol in self.context_memory:
                self.context_memory[symbol].clear()
            if symbol in self.performance_memory:
                self.performance_memory[symbol] = {
                    'buy': {'count': 0, 'success': 0, 'pnl_sum': 0.0, 'confidence_sum': 0.0},
                    'sell': {'count': 0, 'success': 0, 'pnl_sum': 0.0, 'confidence_sum': 0.0},
                    'by_regime': defaultdict(
                        lambda: {'count': 0, 'success': 0, 'pnl_sum': 0.0}
                    ),
                    'total_signals': 0,
                    'successful_signals': 0,
                    'avg_pnl': 0.0,
                    'last_update': None
                }
            logger.info(f"Memory reset for symbol {symbol}")
        else:
            self.signal_memory.clear()
            self.context_memory.clear()
            self.performance_memory.clear()
            logger.info("All memory reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the meta-memory system.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_symbols': len(self.signal_memory),
            'total_signals': sum(len(signals) for signals in self.signal_memory.values()),
            'signals_with_outcomes': sum(
                sum(1 for s in signals if s['outcome'] is not None)
                for signals in self.signal_memory.values()
            ),
            'symbols': {},
            'regimes': set()
        }
        
        # Collect symbol-specific statistics
        for symbol in self.signal_memory:
            symbol_stats = {
                'signal_count': len(self.signal_memory[symbol]),
                'outcome_count': sum(1 for s in self.signal_memory[symbol] if s['outcome'] is not None),
                'regimes': set(s['context'].get('market_regime', 'unknown') for s in self.signal_memory[symbol]),
                'performance': {}
            }
            
            # Add performance statistics if available
            if symbol in self.performance_memory:
                perf = self.performance_memory[symbol]
                symbol_stats['performance'] = {
                    'total_signals': perf['total_signals'],
                    'success_rate': perf['successful_signals'] / perf['total_signals'] if perf['total_signals'] > 0 else 0,
                    'avg_pnl': perf['avg_pnl'],
                    'buy_success_rate': perf['buy']['success'] / perf['buy']['count'] if perf['buy']['count'] > 0 else 0,
                    'sell_success_rate': perf['sell']['success'] / perf['sell']['count'] if perf['sell']['count'] > 0 else 0,
                    'regimes': {
                        regime: {
                            'count': stats['count'],
                            'success_rate': stats['success'] / stats['count'] if stats['count'] > 0 else 0,
                            'avg_pnl': stats['pnl_sum'] / stats['count'] if stats['count'] > 0 else 0
                        }
                        for regime, stats in perf['by_regime'].items()
                        if stats['count'] > 0
                    }
                }
            
            stats['symbols'][symbol] = symbol_stats
            stats['regimes'].update(symbol_stats['regimes'])
        
        stats['regimes'] = list(stats['regimes'])
        
        return stats 