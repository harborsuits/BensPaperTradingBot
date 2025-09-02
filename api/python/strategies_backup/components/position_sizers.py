"""
Position Sizer Components

Implementation of various position sizing components for the modular strategy system.
These components determine position sizes based on risk, volatility, and other factors.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

from trading_bot.strategies.base_strategy import SignalType
from trading_bot.strategies.modular_strategy_system import (
    StrategyComponent, ComponentType, PositionSizerComponent
)

logger = logging.getLogger(__name__)

class FixedRiskPositionSizer(PositionSizerComponent):
    """Sizes positions based on fixed risk percentage."""
    
    def __init__(self, 
                component_id: Optional[str] = None,
                risk_per_trade: float = 1.0,
                max_position_size: float = 10.0,
                atr_period: int = 14,
                atr_multiplier: float = 2.0):
        """
        Initialize fixed risk position sizer
        
        Args:
            component_id: Unique component ID
            risk_per_trade: Risk percentage per trade (% of account)
            max_position_size: Maximum position size (% of account)
            atr_period: ATR period for stop loss calculation
            atr_multiplier: ATR multiplier for stop loss
        """
        super().__init__(component_id)
        self.parameters = {
            'risk_per_trade': risk_per_trade,
            'max_position_size': max_position_size,
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier
        }
        self.description = f"Fixed Risk Sizer ({risk_per_trade:.1f}% risk)"
    
    def calculate_position_sizes(self, signals: Dict[str, SignalType], 
                              data: Dict[str, pd.DataFrame], 
                              context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate position sizes based on fixed risk
        
        Args:
            signals: Trading signals
            data: Market data
            context: Processing context
            
        Returns:
            Dictionary of symbol -> position size (as % of account)
        """
        position_sizes = {}
        
        # Get account value and parameters
        account_value = context.get('account_value', 0)
        if account_value <= 0:
            logger.warning("Account value is zero or negative, can't size positions")
            return position_sizes
        
        risk_pct = self.parameters['risk_per_trade']
        max_size_pct = self.parameters['max_position_size']
        atr_period = self.parameters['atr_period']
        atr_multiplier = self.parameters['atr_multiplier']
        
        # Calculate dollar risk amount
        risk_amount = account_value * (risk_pct / 100)
        
        for symbol, signal in signals.items():
            # Skip FLAT signals
            if signal == SignalType.FLAT:
                position_sizes[symbol] = 0.0
                continue
            
            # Get market data for this symbol
            if symbol not in data:
                position_sizes[symbol] = 0.0
                continue
            
            df = data[symbol]
            if len(df) < atr_period + 1:
                position_sizes[symbol] = 0.0
                continue
            
            # Calculate ATR if not available
            if 'atr' not in df.columns:
                # Calculate True Range
                high_low = df['high'] - df['low']
                high_close_prev = abs(df['high'] - df['close'].shift(1))
                low_close_prev = abs(df['low'] - df['close'].shift(1))
                tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                
                # Calculate ATR
                df['atr'] = tr.rolling(window=atr_period).mean()
            
            # Get latest price and ATR
            last_price = df['close'].iloc[-1]
            last_atr = df['atr'].iloc[-1]
            
            if pd.isna(last_atr) or last_atr <= 0 or pd.isna(last_price) or last_price <= 0:
                position_sizes[symbol] = 0.0
                continue
            
            # Calculate stop loss distance
            stop_distance = last_atr * atr_multiplier
            
            # Calculate position size
            shares = risk_amount / stop_distance
            position_value = shares * last_price
            position_pct = (position_value / account_value) * 100
            
            # Cap at maximum size
            position_pct = min(position_pct, max_size_pct)
            
            # Store position size
            position_sizes[symbol] = position_pct
        
        return position_sizes

class VolatilityAdjustedPositionSizer(PositionSizerComponent):
    """Sizes positions based on volatility."""
    
    def __init__(self, 
                component_id: Optional[str] = None,
                base_position_size: float = 5.0,
                volatility_lookback: int = 20,
                min_position_size: float = 1.0,
                max_position_size: float = 10.0,
                inverse_volatility: bool = True):
        """
        Initialize volatility-adjusted position sizer
        
        Args:
            component_id: Unique component ID
            base_position_size: Base position size (% of account)
            volatility_lookback: Lookback period for volatility calculation
            min_position_size: Minimum position size (% of account)
            max_position_size: Maximum position size (% of account)
            inverse_volatility: True to reduce size for higher volatility
        """
        super().__init__(component_id)
        self.parameters = {
            'base_position_size': base_position_size,
            'volatility_lookback': volatility_lookback,
            'min_position_size': min_position_size,
            'max_position_size': max_position_size,
            'inverse_volatility': inverse_volatility
        }
        self.description = f"Volatility-Adjusted Sizer (±{max_position_size:.1f}%)"
    
    def calculate_position_sizes(self, signals: Dict[str, SignalType], 
                              data: Dict[str, pd.DataFrame], 
                              context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate position sizes based on volatility
        
        Args:
            signals: Trading signals
            data: Market data
            context: Processing context
            
        Returns:
            Dictionary of symbol -> position size (as % of account)
        """
        position_sizes = {}
        
        # Get parameters
        base_size = self.parameters['base_position_size']
        lookback = self.parameters['volatility_lookback']
        min_size = self.parameters['min_position_size']
        max_size = self.parameters['max_position_size']
        inverse = self.parameters['inverse_volatility']
        
        # Store volatility values for normalization
        volatilities = {}
        
        # First pass: calculate volatility for all symbols with signals
        for symbol, signal in signals.items():
            # Skip FLAT signals
            if signal == SignalType.FLAT:
                position_sizes[symbol] = 0.0
                continue
            
            # Get market data for this symbol
            if symbol not in data:
                position_sizes[symbol] = 0.0
                continue
            
            df = data[symbol]
            if len(df) < lookback + 1:
                position_sizes[symbol] = 0.0
                continue
            
            # Calculate volatility (standard deviation of returns)
            returns = df['close'].pct_change().iloc[-lookback:]
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            if pd.isna(volatility) or volatility <= 0:
                position_sizes[symbol] = 0.0
                continue
            
            volatilities[symbol] = volatility
        
        if not volatilities:
            return position_sizes
        
        # Normalize volatilities
        min_vol = min(volatilities.values())
        max_vol = max(volatilities.values())
        vol_range = max_vol - min_vol
        
        for symbol, vol in volatilities.items():
            # Normalize to 0-1 range
            if vol_range > 0:
                norm_vol = (vol - min_vol) / vol_range
            else:
                norm_vol = 0.5  # If all are the same
            
            # Inverse if needed
            if inverse:
                norm_vol = 1 - norm_vol
            
            # Calculate position size
            size = base_size * (0.5 + norm_vol)  # Scale around base
            
            # Ensure it's within limits
            size = max(min_size, min(size, max_size))
            
            # Store position size
            position_sizes[symbol] = size
        
        return position_sizes

class KellyPositionSizer(PositionSizerComponent):
    """Sizes positions based on Kelly criterion."""
    
    def __init__(self, 
                component_id: Optional[str] = None,
                win_rate_lookback: int = 50,
                payoff_lookback: int = 50,
                max_position_size: float = 10.0,
                kelly_fraction: float = 0.5):
        """
        Initialize Kelly-based position sizer
        
        Args:
            component_id: Unique component ID
            win_rate_lookback: Lookback period for win rate calculation
            payoff_lookback: Lookback period for payoff ratio calculation
            max_position_size: Maximum position size (% of account)
            kelly_fraction: Fraction of full Kelly to use (0-1)
        """
        super().__init__(component_id)
        self.parameters = {
            'win_rate_lookback': win_rate_lookback,
            'payoff_lookback': payoff_lookback,
            'max_position_size': max_position_size,
            'kelly_fraction': kelly_fraction
        }
        self.description = f"Kelly Sizer ({kelly_fraction:.1f}x Kelly)"
        
        # Store historical trades
        self.historical_trades = {}
    
    def calculate_position_sizes(self, signals: Dict[str, SignalType], 
                              data: Dict[str, pd.DataFrame], 
                              context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate position sizes based on Kelly criterion
        
        Args:
            signals: Trading signals
            data: Market data
            context: Processing context
            
        Returns:
            Dictionary of symbol -> position size (as % of account)
        """
        position_sizes = {}
        
        # Get parameters
        win_rate_lookback = self.parameters['win_rate_lookback']
        payoff_lookback = self.parameters['payoff_lookback']
        max_size = self.parameters['max_position_size']
        kelly_fraction = self.parameters['kelly_fraction']
        
        # Get trade history from context
        trade_history = context.get('trade_history', {})
        
        for symbol, signal in signals.items():
            # Skip FLAT signals
            if signal == SignalType.FLAT:
                position_sizes[symbol] = 0.0
                continue
            
            # Get trade history for this symbol
            symbol_history = trade_history.get(symbol, [])
            
            # Need enough history
            if len(symbol_history) < max(win_rate_lookback, payoff_lookback):
                # Fall back to default size (half of max)
                position_sizes[symbol] = max_size / 2
                continue
            
            # Calculate win rate
            recent_trades = symbol_history[-win_rate_lookback:]
            wins = sum(1 for trade in recent_trades if trade['profit_pct'] > 0)
            win_rate = wins / len(recent_trades)
            
            # Calculate payoff ratio
            recent_trades = symbol_history[-payoff_lookback:]
            wins = [trade['profit_pct'] for trade in recent_trades if trade['profit_pct'] > 0]
            losses = [abs(trade['profit_pct']) for trade in recent_trades if trade['profit_pct'] < 0]
            
            if not wins or not losses:
                # Can't calculate payoff, use default
                position_sizes[symbol] = max_size / 2
                continue
            
            avg_win = sum(wins) / len(wins)
            avg_loss = sum(losses) / len(losses)
            
            if avg_loss <= 0:
                # Invalid data, use default
                position_sizes[symbol] = max_size / 2
                continue
            
            payoff_ratio = avg_win / avg_loss
            
            # Calculate Kelly percentage
            kelly_pct = win_rate - ((1 - win_rate) / payoff_ratio)
            
            # Apply Kelly fraction and bounds
            position_pct = kelly_pct * kelly_fraction * 100  # Convert to percentage
            position_pct = max(0, min(position_pct, max_size))
            
            # Store position size
            position_sizes[symbol] = position_pct
        
        return position_sizes

class EqualWeightPositionSizer(PositionSizerComponent):
    """Sizes positions with equal weight."""
    
    def __init__(self, 
                component_id: Optional[str] = None,
                position_size: float = 5.0,
                max_positions: int = 10,
                max_total_exposure: float = 50.0):
        """
        Initialize equal weight position sizer
        
        Args:
            component_id: Unique component ID
            position_size: Position size per trade (% of account)
            max_positions: Maximum number of positions
            max_total_exposure: Maximum total exposure (% of account)
        """
        super().__init__(component_id)
        self.parameters = {
            'position_size': position_size,
            'max_positions': max_positions,
            'max_total_exposure': max_total_exposure
        }
        self.description = f"Equal Weight Sizer ({position_size:.1f}% per position)"
    
    def calculate_position_sizes(self, signals: Dict[str, SignalType], 
                              data: Dict[str, pd.DataFrame], 
                              context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate position sizes with equal weights
        
        Args:
            signals: Trading signals
            data: Market data
            context: Processing context
            
        Returns:
            Dictionary of symbol -> position size (as % of account)
        """
        position_sizes = {}
        
        # Get parameters
        size_per_trade = self.parameters['position_size']
        max_positions = self.parameters['max_positions']
        max_exposure = self.parameters['max_total_exposure']
        
        # Count non-FLAT signals
        active_signals = {s: sig for s, sig in signals.items() if sig != SignalType.FLAT}
        
        # Limit number of positions
        if len(active_signals) > max_positions:
            # Prioritize signals somehow (e.g., by strength or alphabetically)
            # For simplicity, just take the first max_positions
            active_signals = dict(list(active_signals.items())[:max_positions])
        
        # Calculate position size
        actual_size = min(size_per_trade, max_exposure / max(1, len(active_signals)))
        
        # Assign sizes
        for symbol, signal in signals.items():
            if signal == SignalType.FLAT or symbol not in active_signals:
                position_sizes[symbol] = 0.0
            else:
                position_sizes[symbol] = actual_size
        
        return position_sizes

class StrengthAdjustedPositionSizer(PositionSizerComponent):
    """Sizes positions based on signal strength."""
    
    def __init__(self, 
                component_id: Optional[str] = None,
                base_position_size: float = 5.0,
                min_position_size: float = 1.0,
                max_position_size: float = 10.0,
                max_total_exposure: float = 50.0):
        """
        Initialize strength-adjusted position sizer
        
        Args:
            component_id: Unique component ID
            base_position_size: Base position size (% of account)
            min_position_size: Minimum position size (% of account)
            max_position_size: Maximum position size (% of account)
            max_total_exposure: Maximum total exposure (% of account)
        """
        super().__init__(component_id)
        self.parameters = {
            'base_position_size': base_position_size,
            'min_position_size': min_position_size,
            'max_position_size': max_position_size,
            'max_total_exposure': max_total_exposure
        }
        self.description = f"Strength-Adjusted Sizer ({min_position_size:.1f}%-{max_position_size:.1f}%)"
    
    def calculate_position_sizes(self, signals: Dict[str, SignalType], 
                              data: Dict[str, pd.DataFrame], 
                              context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate position sizes based on signal strength
        
        Args:
            signals: Trading signals
            data: Market data
            context: Processing context
            
        Returns:
            Dictionary of symbol -> position size (as % of account)
        """
        position_sizes = {}
        
        # Get parameters
        base_size = self.parameters['base_position_size']
        min_size = self.parameters['min_position_size']
        max_size = self.parameters['max_position_size']
        max_exposure = self.parameters['max_total_exposure']
        
        # Get signal strength from context (if available)
        signal_strengths = context.get('signal_strengths', {})
        
        total_strength = 0
        valid_symbols = []
        
        # First pass: get signal strengths and filter signals
        for symbol, signal in signals.items():
            # Skip FLAT signals
            if signal == SignalType.FLAT:
                position_sizes[symbol] = 0.0
                continue
            
            # Get strength or use default
            strength = signal_strengths.get(symbol, 0.5)  # Default to medium strength
            
            # Adjust strength based on signal type
            if signal == SignalType.SCALE_UP:
                strength = min(1.0, strength * 1.5)  # Boost strength
            elif signal == SignalType.SCALE_DOWN:
                strength = min(1.0, strength * 1.5)  # Boost strength
            
            total_strength += strength
            valid_symbols.append((symbol, strength))
        
        if not valid_symbols:
            return position_sizes
        
        # Second pass: calculate sizes based on relative strength
        total_allocated = 0
        
        for symbol, strength in valid_symbols:
            # Calculate relative strength
            relative_strength = strength / total_strength if total_strength > 0 else 1.0 / len(valid_symbols)
            
            # Calculate size based on strength
            size = base_size * (0.5 + strength)  # Scale around base
            
            # Ensure it's within limits
            size = max(min_size, min(size, max_size))
            
            # Track total allocation
            total_allocated += size
            
            # Store position size
            position_sizes[symbol] = size
        
        # Adjust if over maximum exposure
        if total_allocated > max_exposure:
            scale_factor = max_exposure / total_allocated
            
            for symbol in position_sizes:
                position_sizes[symbol] *= scale_factor
        
        return position_sizes
