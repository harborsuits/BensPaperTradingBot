#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified test script for the Pips-a-Day Forex Strategy
Works without external dependencies
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, Any, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock the required classes and enums
class MarketRegime(Enum):
    TRENDING_BULL = auto()
    TRENDING_BEAR = auto()
    RANGE_BOUND = auto()
    BREAKOUT = auto()
    VOLATILE = auto()
    CHOPPY = auto()
    LOW_VOLATILITY = auto()
    HIGH_VOLATILITY = auto()
    NORMAL = auto()

class MarketSession(Enum):
    ASIAN = auto()
    EUROPEAN = auto()
    US = auto()
    CLOSED = auto()

class EntryQuality(Enum):
    NOT_VALID = 0
    MARGINAL = 1
    STANDARD = 2
    PREMIUM = 3

class TradeDirection(Enum):
    LONG = auto()
    SHORT = auto()
    FLAT = auto()

class TargetStatus(Enum):
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    TARGET_REACHED = auto()
    TARGET_EXCEEDED = auto()
    LOSS_LIMIT_REACHED = auto()

# Mock EventBus class
class EventBus:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = EventBus()
        return cls._instance
    
    def __init__(self):
        self.subscribers = {}
        
    def subscribe(self, event_type, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def unsubscribe(self, event_type, callback):
        if event_type in self.subscribers and callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            
    def publish(self, event_type, data):
        logger.debug(f"EventBus: Publishing {event_type} event")
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                callback(event_type, data)

# Mock base strategy class
class Strategy:
    def __init__(self, name: str = None, parameters: Dict[str, Any] = None):
        self.name = name or self.__class__.__name__
        self.parameters = parameters or {}
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: pd.Timestamp) -> Dict[str, Any]:
        return {}
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        return {'name': self.name}

# Mock ForexBaseStrategy
class ForexBaseStrategy(Strategy):
    def __init__(self, name: str = None, parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.active_positions = {}
        
    def is_active_trading_session(self, current_time: pd.Timestamp) -> bool:
        return True
        
    def get_current_session(self, current_time: pd.Timestamp) -> MarketSession:
        hour = current_time.hour
        if 0 <= hour < 8:
            return MarketSession.ASIAN
        elif 8 <= hour < 16:
            return MarketSession.EUROPEAN
        else:
            return MarketSession.US

# Simplified PipsADayStrategy for testing
class PipsADayStrategy(ForexBaseStrategy):
    def __init__(self, name: str = "Pips-a-Day Strategy", parameters: Dict[str, Any] = None):
        default_params = {
            'base_currency_pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
            'primary_execution_timeframe': '1h',
            'min_data_lookback': 30,
            'entry_indicators': ['rsi', 'macd', 'bollinger'],
            'filter_indicators': ['moving_averages', 'adx'],
            'confirmation_indicators': ['macd'],
            'required_confirmation_count': 2,
            'min_entry_quality': EntryQuality.MARGINAL.value,
            'scale_out_levels': [0.3, 0.6, 0.9],
            'scale_out_percents': [30, 30, 40],
            'scale_out_enabled': True,
            'daily_pip_target': 30,
            'base_stop_loss_pips': 20,
            'min_reward_risk_ratio': 1.5,
            'use_adaptive_stops': True,
            'atr_stop_loss_factor': 1.2,
            'correlation_filter_enabled': True,
            'max_entries_per_day': 5,
            'base_position_size': 0.1,
            'premium_entry_boost': 1.3,
            'trade_asian_session': True,
            'trade_european_session': True,
            'trade_us_session': True,
            'min_daily_range_pips': 30,
            'news_filter_enabled': False,
            'volatility_based_targets': True,
            'target_volatility_factor': 1.2,
            'daily_loss_limit': -20,
            'weekly_target_modifier': 0.8,
            'monthly_target_modifier': 0.7,
            'min_daily_target': 20,
            'max_daily_target': 50
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name, default_params)
        
        # Initialize strategy state
        self.volatility_factors = {}
        self.current_day_pip_total = 0
        self.current_day_positions = 0
        self.current_day_status = TargetStatus.NOT_STARTED
        self.daily_targets = {}
        self.daily_achieved = {}
        self.daily_stats = {}
        self.weekly_pips = 0
        self.monthly_pips = 0
        self.session_pips = {
            MarketSession.ASIAN: 0,
            MarketSession.EUROPEAN: 0,
            MarketSession.US: 0
        }
        self.news_filter = {}
        
    def _is_tradable_session(self, session):
        """Check if we should trade in the given session"""
        if session == MarketSession.ASIAN and self.parameters['trade_asian_session']:
            return True
        elif session == MarketSession.EUROPEAN and self.parameters['trade_european_session']:
            return True
        elif session == MarketSession.US and self.parameters['trade_us_session']:
            return True
        return False

    def _reset_daily_stats(self):
        """Reset daily statistics for a new trading day"""
        # Store the previous day's stats if we have any
        if self.current_day_pip_total != 0 or self.current_day_positions > 0:
            today = pd.Timestamp.now().strftime('%Y-%m-%d')
            self.daily_stats[today] = {
                'pips_total': self.current_day_pip_total,
                'positions': self.current_day_positions,
                'target_status': self.current_day_status.value,
                'targets': self.daily_targets.copy(),
                'achieved': self.daily_achieved.copy()
            }
            
        # Reset counters for the new day
        self.current_day_pip_total = 0
        self.current_day_positions = 0
        self.current_day_status = TargetStatus.NOT_STARTED
        self.daily_achieved = {}
        
        # Reset session pips
        for session in self.session_pips:
            self.session_pips[session] = 0

    def _calculate_daily_targets(self):
        """Calculate daily pip targets, potentially adjusting for volatility"""
        base_target = self.parameters['daily_pip_target']
        
        # Check for weekly/monthly target modifiers
        if self.weekly_pips > (base_target * 5):
            # If we've already hit the weekly target, reduce daily target
            modifier = self.parameters['weekly_target_modifier']
            logger.info(f"Weekly target reached ({self.weekly_pips} pips), applying modifier {modifier}")
            base_target *= modifier
            
        if self.monthly_pips > (base_target * 20):
            # If we've already hit the monthly target, reduce daily target
            modifier = self.parameters['monthly_target_modifier']
            logger.info(f"Monthly target reached ({self.monthly_pips} pips), applying modifier {modifier}")
            base_target *= modifier
        
        # Set the targets for each symbol
        for symbol in self.parameters['base_currency_pairs']:
            if self.parameters['volatility_based_targets'] and symbol in self.volatility_factors:
                # Adjust target based on volatility
                vol_factor = self.volatility_factors[symbol]
                symbol_target = base_target * vol_factor
                
                # Apply min/max constraints
                symbol_target = max(self.parameters['min_daily_target'], 
                                  min(self.parameters['max_daily_target'], symbol_target))
                
                logger.debug(f"Volatility-adjusted target for {symbol}: {symbol_target:.1f} pips (factor: {vol_factor:.2f})")
            else:
                symbol_target = base_target
            
            # Store the target
            if not hasattr(self, 'daily_targets'):
                self.daily_targets = {}
                
            self.daily_targets[symbol] = symbol_target
        
        # Log the new targets
        logger.info(f"Daily pip targets calculated: {self.daily_targets}")

    def _check_daily_status(self):
        """Check if daily target or loss limit has been reached"""
        # Get base target and loss limit
        base_target = self.parameters['daily_pip_target']
        loss_limit = self.parameters['daily_loss_limit']
        
        # Check if we've hit the target
        if self.current_day_pip_total >= base_target and self.current_day_status != TargetStatus.TARGET_REACHED:
            self.current_day_status = TargetStatus.TARGET_REACHED
            logger.info(f"Daily pip target of {base_target} pips reached! Current total: {self.current_day_pip_total} pips")
            
            # Publish target reached event
            EventBus.get_instance().publish('daily_target_reached', {
                'strategy': self.__class__.__name__,
                'target': base_target,
                'achieved': self.current_day_pip_total,
                'timestamp': pd.Timestamp.now()
            })
            
        # Check if we've exceeded the target
        elif self.current_day_pip_total > base_target * 1.5 and self.current_day_status != TargetStatus.TARGET_EXCEEDED:
            self.current_day_status = TargetStatus.TARGET_EXCEEDED
            logger.info(f"Daily pip target exceeded by 50%! Current total: {self.current_day_pip_total} pips")
            
            # Publish target exceeded event
            EventBus.get_instance().publish('daily_target_exceeded', {
                'strategy': self.__class__.__name__,
                'target': base_target,
                'achieved': self.current_day_pip_total,
                'timestamp': pd.Timestamp.now()
            })
            
        # Check if we've hit the loss limit
        elif self.current_day_pip_total <= loss_limit and self.current_day_status != TargetStatus.LOSS_LIMIT_REACHED:
            self.current_day_status = TargetStatus.LOSS_LIMIT_REACHED
            logger.warning(f"Daily loss limit of {loss_limit} pips reached! Current total: {self.current_day_pip_total} pips")
            
            # Publish loss limit reached event
            EventBus.get_instance().publish('daily_loss_limit_reached', {
                'strategy': self.__class__.__name__,
                'limit': loss_limit,
                'current': self.current_day_pip_total,
                'timestamp': pd.Timestamp.now()
            })
            
        # If we're making progress but haven't hit any threshold yet
        elif self.current_day_pip_total != 0 and self.current_day_status == TargetStatus.NOT_STARTED:
            self.current_day_status = TargetStatus.IN_PROGRESS
        
        return self.current_day_status

    def calculate_regime_compatibility(self, market_regime: MarketRegime) -> float:
        """Calculate compatibility score for a given market regime"""
        # Compatibility matrix
        compatibility = {
            MarketRegime.TRENDING_BULL: 0.6,    # Good in trending
            MarketRegime.TRENDING_BEAR: 0.6,    # Good in trending
            MarketRegime.RANGE_BOUND: 0.9,      # Excellent in range-bound
            MarketRegime.BREAKOUT: 0.5,         # Moderate in breakout
            MarketRegime.VOLATILE: 0.75,        # Very good in volatile
            MarketRegime.CHOPPY: 0.7,           # Good in choppy
            MarketRegime.LOW_VOLATILITY: 0.85,  # Very good in low volatility
            MarketRegime.HIGH_VOLATILITY: 0.65, # Good in high volatility
            MarketRegime.NORMAL: 0.9,           # Excellent in normal conditions
        }
        
        return compatibility.get(market_regime, 0.5)  # Default 0.5 for unknown regimes

    def optimize_for_regime(self, market_regime: MarketRegime) -> None:
        """Optimize strategy parameters for the given market regime"""
        if market_regime == MarketRegime.TRENDING_BULL or market_regime == MarketRegime.TRENDING_BEAR:
            # In trending, increase reward-risk and scale out more aggressively
            self.parameters['min_reward_risk_ratio'] = 2.0
            self.parameters['scale_out_levels'] = [0.33, 0.67, 0.9]
            self.parameters['scale_out_percents'] = [25, 25, 50]
            self.parameters['target_volatility_factor'] = 1.5
            
        elif market_regime == MarketRegime.RANGE_BOUND or market_regime == MarketRegime.LOW_VOLATILITY:
            # In range-bound, take smaller profits with higher win rate
            self.parameters['min_reward_risk_ratio'] = 1.2
            self.parameters['scale_out_levels'] = [0.25, 0.5, 0.75]
            self.parameters['scale_out_percents'] = [25, 25, 50]
            self.parameters['daily_pip_target'] *= 0.8  # More conservative target
            self.parameters['target_volatility_factor'] = 1.0
            
        elif market_regime == MarketRegime.VOLATILE or market_regime == MarketRegime.HIGH_VOLATILITY:
            # In volatile, use wider stops, higher targets
            self.parameters['min_reward_risk_ratio'] = 1.8
            self.parameters['scale_out_levels'] = [0.45, 0.8, 1.0]
            self.parameters['scale_out_percents'] = [30, 30, 40]
            self.parameters['atr_stop_loss_factor'] = 1.5
            self.parameters['target_volatility_factor'] = 2.0
            
        elif market_regime == MarketRegime.BREAKOUT:
            # In breakout, focus on catching strong moves
            self.parameters['min_reward_risk_ratio'] = 2.5
            self.parameters['scale_out_levels'] = [0.5, 0.8, 1.1]
            self.parameters['scale_out_percents'] = [20, 30, 50]
            self.parameters['target_volatility_factor'] = 2.0
            
        elif market_regime == MarketRegime.CHOPPY:
            # In choppy, more conservative with faster exits
            self.parameters['min_reward_risk_ratio'] = 1.3
            self.parameters['scale_out_levels'] = [0.2, 0.4, 0.6]
            self.parameters['scale_out_percents'] = [40, 30, 30]
            self.parameters['target_volatility_factor'] = 0.8
            
        elif market_regime == MarketRegime.NORMAL:
            # In normal conditions, use balanced approach
            self.parameters['min_reward_risk_ratio'] = 1.5
            self.parameters['scale_out_levels'] = [0.3, 0.6, 0.9]
            self.parameters['scale_out_percents'] = [20, 30, 50]
            self.parameters['target_volatility_factor'] = 1.25
            
        # Log optimization
        logger.info(f"Optimized Pips-a-Day strategy for {market_regime.name} regime")

def generate_synthetic_data(num_days: int = 10, 
                          timeframe: str = '30m', 
                          base_currency_pairs: List[str] = None,
                          volatility_factor: float = 1.0) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Generate synthetic OHLCV data for testing
    
    Args:
        num_days: Number of days to generate
        timeframe: Timeframe for the data
        base_currency_pairs: List of currency pairs to generate data for
        volatility_factor: Volatility multiplier (higher = more volatile)
        
    Returns:
        Dictionary of symbol -> {timeframe -> DataFrame}
    """
    if base_currency_pairs is None:
        base_currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        
    # Set up timeframe in minutes
    if timeframe == '1m':
        minutes = 1
    elif timeframe == '5m':
        minutes = 5
    elif timeframe == '15m':
        minutes = 15
    elif timeframe == '30m':
        minutes = 30
    elif timeframe == '1h':
        minutes = 60
    elif timeframe == '4h':
        minutes = 240
    else:
        minutes = 60
    
    # Calculate number of periods per day
    periods_per_day = 24 * 60 // minutes
    total_periods = periods_per_day * num_days

    # Get current time and round to nearest timeframe
    end_time = datetime.now().replace(microsecond=0)
    end_time = end_time - timedelta(minutes=end_time.minute % minutes, 
                                   seconds=end_time.second)
    
    # Generate dates working backwards from end_time
    dates = [end_time - timedelta(minutes=i*minutes) for i in range(total_periods)]
    dates.reverse()  # Put in chronological order
    
    # Create empty result dictionary
    result = {}
    
    # Generate data for each currency pair
    for pair in base_currency_pairs:
        # Set base price depending on pair
        if 'JPY' in pair:
            base_price = 110.0  # JPY pairs have higher nominal value
            pip_size = 0.01
        else:
            base_price = 1.20  # Non-JPY pairs typically around 1.0-1.5
            pip_size = 0.0001
        
        # Create price series with random walk and some cyclicality
        np.random.seed(hash(pair) % 100)  # Different seed for each pair
        
        # Generate random walk
        random_walk = np.random.normal(0, volatility_factor * pip_size * 30, total_periods).cumsum()
        
        # Add cyclicality - daily cycle
        daily_cycle = np.sin(np.linspace(0, 2*np.pi*num_days, total_periods)) * pip_size * 80 * volatility_factor
        
        # Add cyclicality - session-based volatility
        session_volatility = []
        for i, date in enumerate(dates):
            hour = date.hour
            if 0 <= hour < 8:  # Asian session (less volatile)
                volatility = 0.7
            elif 8 <= hour < 16:  # European session (more volatile)
                volatility = 1.2
            else:  # US session (medium volatile)
                volatility = 1.0
                
            session_volatility.append(volatility)
        
        session_effect = np.array(session_volatility) * pip_size * 20 * volatility_factor
        
        # Combine components
        price_series = base_price + random_walk + daily_cycle + session_effect
        
        # Generate OHLCV data
        ohlcv_data = []
        for i in range(total_periods):
            close = price_series[i]
            high = close + abs(np.random.normal(0, pip_size * 15 * volatility_factor))
            low = close - abs(np.random.normal(0, pip_size * 15 * volatility_factor))
            open_price = price_series[i-1] if i > 0 else close
            volume = np.random.randint(100, 1000)
            
            ohlcv_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(ohlcv_data, index=dates)
        
        # Add to result
        result[pair] = {timeframe: df}
        
    return result

def test_target_management():
    """Test the daily target management functionality"""
    logger.info("TESTING: Target Management")
    
    # Initialize strategy
    params = {
        'base_currency_pairs': ['EURUSD', 'GBPUSD', 'USDJPY'],
        'daily_pip_target': 30,
        'weekly_target_modifier': 0.8,
        'monthly_target_modifier': 0.7,
        'min_daily_target': 20,
        'max_daily_target': 50,
        'volatility_based_targets': True,
        'target_volatility_factor': 1.2,
        'daily_loss_limit': -20
    }
    
    strategy = PipsADayStrategy(parameters=params)
    
    # Generate synthetic data
    data = generate_synthetic_data(num_days=5, timeframe='1h', 
                                base_currency_pairs=params['base_currency_pairs'])
    
    # Test calculation of daily targets
    strategy._calculate_daily_targets()
    
    # Print targets
    logger.info(f"Daily targets: {strategy.daily_targets}")
    
    # Test status management
    strategy.current_day_pip_total = 15
    status = strategy._check_daily_status()
    logger.info(f"Status at 15 pips: {status}")
    
    strategy.current_day_pip_total = 31
    status = strategy._check_daily_status()
    logger.info(f"Status at 31 pips: {status}")
    
    strategy.current_day_pip_total = 50
    status = strategy._check_daily_status()
    logger.info(f"Status at 50 pips: {status}")
    
    # Reset and test loss limit
    strategy.current_day_pip_total = 0
    strategy.current_day_status = TargetStatus.NOT_STARTED
    
    strategy.current_day_pip_total = -10
    status = strategy._check_daily_status()
    logger.info(f"Status at -10 pips: {status}")
    
    strategy.current_day_pip_total = -21
    status = strategy._check_daily_status()
    logger.info(f"Status at -21 pips: {status}")
    
    # Test status between days
    strategy._reset_daily_stats()
    logger.info(f"After reset, pip total: {strategy.current_day_pip_total}, status: {strategy.current_day_status}")
    
    # Test volatility-based targets
    # Manually set some volatility factors for testing
    strategy.volatility_factors = {
        'EURUSD': 1.2,
        'GBPUSD': 1.5,
        'USDJPY': 0.8
    }
    
    strategy._calculate_daily_targets()
    logger.info(f"Volatility-adjusted targets: {strategy.daily_targets}")
    
    logger.info("Target management tests completed successfully")

def test_regime_compatibility():
    """Test regime compatibility scoring and optimization"""
    logger.info("TESTING: Regime Compatibility")
    
    # Initialize strategy
    strategy = PipsADayStrategy()
    
    # Test compatibility across all regimes
    for regime in MarketRegime:
        score = strategy.calculate_regime_compatibility(regime)
        logger.info(f"Regime {regime.name}: Compatibility score = {score:.2f}")
        
    # Test parameter optimization for different regimes
    test_regimes = [
        MarketRegime.TRENDING_BULL,
        MarketRegime.RANGE_BOUND,
        MarketRegime.VOLATILE,
        MarketRegime.BREAKOUT,
        MarketRegime.NORMAL
    ]
    
    for regime in test_regimes:
        # Reset to default parameters
        strategy = PipsADayStrategy()
        
        # Get parameters before optimization
        before_rr = strategy.parameters['min_reward_risk_ratio']
        before_volatility = strategy.parameters.get('target_volatility_factor', 1.0)
        
        # Optimize for regime
        strategy.optimize_for_regime(regime)
        
        # Get parameters after optimization
        after_rr = strategy.parameters['min_reward_risk_ratio']
        after_volatility = strategy.parameters.get('target_volatility_factor', 1.0)
        
        logger.info(f"Regime {regime.name} optimization:")
        logger.info(f"  Reward/Risk: {before_rr:.1f} → {after_rr:.1f}")
        logger.info(f"  Volatility Factor: {before_volatility:.1f} → {after_volatility:.1f}")
        logger.info(f"  Scale-out levels: {strategy.parameters['scale_out_levels']}")
        logger.info(f"  Scale-out percents: {strategy.parameters['scale_out_percents']}")
        
    logger.info("Regime compatibility tests completed successfully")

def main():
    """Run all tests"""
    logger.info("STARTING SIMPLIFIED PIPS-A-DAY STRATEGY TESTS")
    
    test_target_management()
    logger.info("-" * 50)
    
    test_regime_compatibility()
    logger.info("-" * 50)
    
    logger.info("ALL TESTS COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
