#!/usr/bin/env python
"""
Enhanced Position Sizing System

This module implements advanced position sizing strategies that incorporate:
1. Kelly criterion with fractional Kelly adjustments
2. Volatility-adjusted position sizing
3. Regime-aware risk adjustments
4. Portfolio heat optimization
5. Dynamic trade size calculation based on market conditions
"""

import logging
import numpy as np
import math
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from trading_bot.core.event_bus import EventBus, Event, get_global_event_bus
from trading_bot.core.constants import EventType

logger = logging.getLogger(__name__)


class EnhancedPositionSizer:
    """
    Advanced position sizing system that dynamically adjusts position sizes
    based on multiple factors including volatility, market regime, and 
    portfolio heat.
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the enhanced position sizer.
        
        Args:
            event_bus: Event bus for events
            config: Configuration dictionary
        """
        self.event_bus = event_bus or get_global_event_bus()
        self.config = config or {}
        
        # Kelly criterion settings
        self.use_kelly = self.config.get("use_kelly", True)
        self.kelly_fraction = self.config.get("kelly_fraction", 0.3)  # Fractional Kelly (conservative)
        self.min_kelly_data_points = self.config.get("min_kelly_data_points", 20)
        self.max_kelly_allocation = self.config.get("max_kelly_allocation", 0.1)  # 10% cap
        
        # Volatility settings
        self.volatility_lookback = self.config.get("volatility_lookback", 20)
        self.volatility_adjustment_factor = self.config.get("volatility_adjustment_factor", 1.0)
        self.vol_target = self.config.get("vol_target", 0.15)  # 15% annualized vol target
        self.vol_measure = self.config.get("vol_measure", "atr")  # 'atr', 'stdev', or 'parkinson'
        
        # Risk parameter settings
        self.default_risk_per_trade = self.config.get("default_risk_per_trade", 0.01)  # 1% risk
        self.max_risk_per_trade = self.config.get("max_risk_per_trade", 0.03)  # 3% max risk
        self.min_risk_per_trade = self.config.get("min_risk_per_trade", 0.0025)  # 0.25% min risk
        
        # Portfolio heat settings
        self.max_portfolio_heat = self.config.get("max_portfolio_heat", 0.3)  # 30% max heat
        self.heat_per_strategy = self.config.get("heat_per_strategy", {})  # Strategy-specific heat
        self.current_heat = 0.0
        
        # Snowball risk configuration (increasing size with profits)
        self.use_snowball = self.config.get("use_snowball", True)
        self.snowball_threshold = self.config.get("snowball_threshold", 0.1)  # 10% profit before snowball
        self.snowball_factor = self.config.get("snowball_factor", 0.3)  # Use 30% of excess profit for sizing
        self.snowball_cap = self.config.get("snowball_cap", 3.0)  # Cap at 3x original size
        
        # Market regime adjustments
        self.regime_adjustments = self.config.get("regime_adjustments", {
            "trending": 1.2,    # Increase size in trending markets
            "ranging": 0.8,     # Decrease size in ranging markets
            "volatile": 0.6,    # Significantly decrease in volatile markets
            "low_volatility": 1.0  # Normal sizing in low volatility
        })
        
        # Strategy-specific adjustments
        self.strategy_adjustments = self.config.get("strategy_adjustments", {})
        
        # Trade metrics for adaptive sizing
        self.strategy_metrics = {}
        self.symbol_metrics = {}
        
        # Cache recent calculations
        self.calculation_cache = {}
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        logger.info("Enhanced Position Sizer initialized")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events for position sizing adjustments"""
        self.event_bus.subscribe(EventType.MARKET_REGIME_CHANGED, self._handle_regime_change)
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, self._handle_trade_executed)
        self.event_bus.subscribe(EventType.TRADE_CLOSED, self._handle_trade_closed)
        self.event_bus.subscribe(EventType.PORTFOLIO_EXPOSURE_UPDATED, self._handle_portfolio_update)
    
    def _handle_regime_change(self, event: Event):
        """
        Handle market regime change events
        
        Args:
            event: Market regime change event
        """
        symbol = event.data.get('symbol')
        regime = event.data.get('current_regime')
        
        if symbol and regime:
            # Clear cache for this symbol
            cache_keys = [k for k in self.calculation_cache if k.startswith(f"{symbol}:")]
            for key in cache_keys:
                self.calculation_cache.pop(key, None)
                
            logger.info(f"Position sizing adjusted for {symbol} based on regime change to {regime}")
    
    def _handle_trade_executed(self, event: Event):
        """
        Handle trade executed events
        
        Args:
            event: Trade executed event
        """
        strategy = event.data.get('strategy')
        symbol = event.data.get('symbol')
        position_size = event.data.get('position_size', 0)
        account_value = event.data.get('account_value', 0)
        
        if strategy and symbol and account_value > 0:
            # Update heat
            if position_size > 0:
                heat = position_size / account_value
                self.current_heat += heat
                
                # Update strategy-specific heat
                if strategy not in self.heat_per_strategy:
                    self.heat_per_strategy[strategy] = 0
                self.heat_per_strategy[strategy] += heat
                
                logger.debug(f"Heat increased by {heat:.2%} to {self.current_heat:.2%}")
    
    def _handle_trade_closed(self, event: Event):
        """
        Handle trade closed events
        
        Args:
            event: Trade closed event
        """
        strategy = event.data.get('strategy')
        symbol = event.data.get('symbol')
        pnl = event.data.get('pnl', 0)
        win = event.data.get('win', False)
        position_size = event.data.get('position_size', 0)
        account_value = event.data.get('account_value', 0)
        
        if strategy and symbol:
            # Update strategy metrics
            if strategy not in self.strategy_metrics:
                self.strategy_metrics[strategy] = {
                    'trades': 0,
                    'wins': 0,
                    'total_pnl': 0,
                    'win_pnl': 0,
                    'loss_pnl': 0
                }
                
            metrics = self.strategy_metrics[strategy]
            metrics['trades'] += 1
            if win:
                metrics['wins'] += 1
                metrics['win_pnl'] += pnl
            else:
                metrics['loss_pnl'] += pnl
            metrics['total_pnl'] += pnl
            
            # Update symbol metrics
            symbol_key = f"{strategy}:{symbol}"
            if symbol_key not in self.symbol_metrics:
                self.symbol_metrics[symbol_key] = {
                    'trades': 0,
                    'wins': 0,
                    'total_pnl': 0,
                    'win_pnl': 0,
                    'loss_pnl': 0
                }
                
            s_metrics = self.symbol_metrics[symbol_key]
            s_metrics['trades'] += 1
            if win:
                s_metrics['wins'] += 1
                s_metrics['win_pnl'] += pnl
            else:
                s_metrics['loss_pnl'] += pnl
            s_metrics['total_pnl'] += pnl
            
            # Update heat
            if position_size > 0 and account_value > 0:
                heat = position_size / account_value
                self.current_heat = max(0, self.current_heat - heat)
                
                # Update strategy-specific heat
                if strategy in self.heat_per_strategy:
                    self.heat_per_strategy[strategy] = max(0, self.heat_per_strategy[strategy] - heat)
                
                logger.debug(f"Heat decreased by {heat:.2%} to {self.current_heat:.2%}")
            
            # Clear cache
            cache_key = f"{symbol}:{strategy}"
            if cache_key in self.calculation_cache:
                del self.calculation_cache[cache_key]
    
    def _handle_portfolio_update(self, event: Event):
        """
        Handle portfolio update events
        
        Args:
            event: Portfolio update event
        """
        # Recalculate total portfolio heat if needed
        if 'positions' in event.data:
            positions = event.data.get('positions', [])
            account_value = event.data.get('account_value', 0)
            
            if account_value > 0:
                total_heat = sum(p.get('position_value', 0) for p in positions) / account_value
                self.current_heat = total_heat
                
                # Recalculate strategy heat
                strategy_heat = {}
                for position in positions:
                    strategy = position.get('strategy')
                    if strategy:
                        if strategy not in strategy_heat:
                            strategy_heat[strategy] = 0
                        strategy_heat[strategy] += position.get('position_value', 0) / account_value
                
                self.heat_per_strategy = strategy_heat
    
    def calculate_position_size(
        self,
        account_value: float,
        symbol: str,
        strategy: str,
        entry_price: float,
        stop_price: Optional[float] = None,
        market_data: Optional[Dict[str, Any]] = None,
        volatility: Optional[float] = None,
        regime: Optional[str] = None,
        strategy_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate position size with adaptive adjustments.
        
        Args:
            account_value: Current account value
            symbol: Trading symbol
            strategy: Strategy name
            entry_price: Entry price
            stop_price: Optional stop price
            market_data: Optional market data dictionary
            volatility: Optional volatility measure (if None, will be calculated)
            regime: Optional market regime (if None, will be determined)
            strategy_metrics: Optional strategy performance metrics
            
        Returns:
            Dictionary with position size info including:
            - position_size: Recommended position size in currency
            - position_units: Recommended position size in units
            - risk_amount: Risk amount in currency
            - risk_percent: Risk percentage
            - factors: Dictionary of adjustment factors applied
        """
        cache_key = f"{symbol}:{strategy}:{entry_price}:{stop_price}"
        if cache_key in self.calculation_cache:
            # Use cached calculation if within 5 minutes
            cached = self.calculation_cache[cache_key]
            if datetime.now() - cached['timestamp'] < timedelta(minutes=5):
                return cached['result']
        
        # Base risk percentage from configuration
        base_risk = self.default_risk_per_trade
        
        # Get metrics for this strategy and symbol
        if strategy_metrics is None:
            if strategy in self.strategy_metrics:
                strategy_metrics = self.strategy_metrics[strategy]
        
        # Initialize adjustment factors
        factors = {
            'kelly': 1.0,
            'volatility': 1.0,
            'regime': 1.0,
            'heat': 1.0,
            'snowball': 1.0,
            'strategy_specific': 1.0
        }
        
        # 1. Kelly Criterion adjustment
        if self.use_kelly and strategy_metrics and strategy_metrics.get('trades', 0) >= self.min_kelly_data_points:
            kelly_factor = self._calculate_kelly_factor(strategy_metrics)
            factors['kelly'] = min(kelly_factor, self.max_kelly_allocation) / base_risk
        
        # 2. Volatility adjustment
        vol_factor = 1.0
        if volatility is None and market_data:
            volatility = self._calculate_volatility(market_data, self.vol_measure)
        
        if volatility:
            # Inverse relationship - higher volatility means smaller position
            if self.vol_target > 0:
                vol_factor = self.vol_target / volatility
            factors['volatility'] = min(max(vol_factor * self.volatility_adjustment_factor, 0.5), 2.0)
        
        # 3. Market regime adjustment
        if regime is None and market_data:
            regime = market_data.get('regime', 'unknown')
        
        if regime and regime in self.regime_adjustments:
            factors['regime'] = self.regime_adjustments[regime]
        
        # 4. Portfolio heat adjustment
        heat_remaining = max(0, self.max_portfolio_heat - self.current_heat)
        strategy_max_heat = self.heat_per_strategy.get(strategy, 0.1)  # Default 10% per strategy
        strategy_current_heat = self.heat_per_strategy.get(strategy, 0)
        strategy_heat_remaining = max(0, strategy_max_heat - strategy_current_heat)
        
        # Limit by both overall heat and strategy-specific heat
        heat_limit = min(heat_remaining, strategy_heat_remaining)
        if heat_limit < base_risk:
            factors['heat'] = heat_limit / base_risk
        
        # 5. Snowball adjustment
        if self.use_snowball and strategy in self.strategy_metrics:
            s_metrics = self.strategy_metrics[strategy]
            profit_pct = s_metrics.get('total_pnl', 0) / account_value
            
            if profit_pct > self.snowball_threshold:
                # Calculate snowball factor based on excess profit
                excess_profit = profit_pct - self.snowball_threshold
                snowball_multiplier = 1.0 + (excess_profit * self.snowball_factor)
                snowball_multiplier = min(snowball_multiplier, self.snowball_cap)
                factors['snowball'] = snowball_multiplier
        
        # 6. Strategy-specific adjustment
        if strategy in self.strategy_adjustments:
            factors['strategy_specific'] = self.strategy_adjustments[strategy]
        
        # Calculate combined adjustment
        combined_factor = np.prod(list(factors.values()))
        
        # Calculate adjusted risk percentage
        adjusted_risk = base_risk * combined_factor
        
        # Apply risk limits
        adjusted_risk = min(max(adjusted_risk, self.min_risk_per_trade), self.max_risk_per_trade)
        
        # Calculate risk amount in currency
        risk_amount = account_value * adjusted_risk
        
        # Calculate position size based on stop distance or volatility
        position_size = 0
        position_units = 0
        
        if stop_price is not None and entry_price != stop_price:
            # Calculate based on stop loss
            risk_per_unit = abs(entry_price - stop_price)
            position_units = risk_amount / risk_per_unit
            position_size = position_units * entry_price
        elif volatility:
            # Calculate based on volatility
            risk_per_unit = entry_price * volatility
            position_units = risk_amount / risk_per_unit
            position_size = position_units * entry_price
        else:
            # Fallback method
            position_size = account_value * adjusted_risk
            position_units = position_size / entry_price
        
        # Create result dictionary
        result = {
            'position_size': position_size,
            'position_units': position_units,
            'risk_amount': risk_amount,
            'risk_percent': adjusted_risk,
            'factors': factors,
            'combined_factor': combined_factor,
            'base_risk': base_risk
        }
        
        # Cache the calculation
        self.calculation_cache[cache_key] = {
            'timestamp': datetime.now(),
            'result': result
        }
        
        return result
    
    def _calculate_kelly_factor(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate Kelly Criterion factor based on strategy metrics.
        
        Args:
            metrics: Strategy performance metrics
            
        Returns:
            Kelly allocation factor (0.0-1.0)
        """
        trades = metrics.get('trades', 0)
        if trades == 0:
            return 0.0
        
        wins = metrics.get('wins', 0)
        win_rate = wins / trades
        
        if win_rate == 0:
            return 0.0
        
        # Calculate average win and loss
        avg_win = metrics.get('win_pnl', 0) / max(1, wins)
        avg_loss = abs(metrics.get('loss_pnl', 0)) / max(1, trades - wins)
        
        if avg_loss == 0:
            return self.max_kelly_allocation  # Avoid division by zero
        
        # Calculate win/loss ratio
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula: f* = p - (1-p)/r where p=win rate, r=win/loss ratio
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Apply fractional Kelly for conservatism
        fractional_kelly = kelly * self.kelly_fraction
        
        # Ensure non-negative and cap at max allocation
        return max(0, min(fractional_kelly, self.max_kelly_allocation))
    
    def _calculate_volatility(self, market_data: Dict[str, Any], method: str = 'atr') -> float:
        """
        Calculate volatility from market data.
        
        Args:
            market_data: Market data dictionary
            method: Volatility calculation method ('atr', 'stdev', or 'parkinson')
            
        Returns:
            Volatility as a decimal (e.g., 0.15 for 15%)
        """
        if 'atr' in market_data and method == 'atr':
            # Use ATR normalized by price
            atr = market_data.get('atr', 0)
            price = market_data.get('close', 1)
            return atr / price
        
        # Use historical close prices if available
        if 'ohlc' in market_data:
            ohlc = market_data['ohlc']
            if not ohlc:
                return 0.15  # Default if no data
            
            if method == 'stdev':
                # Standard deviation of returns
                if len(ohlc) < 2:
                    return 0.15
                
                closes = [bar['close'] for bar in ohlc]
                returns = np.diff(np.log(closes))
                return np.std(returns) * np.sqrt(252)  # Annualized
            
            elif method == 'parkinson':
                # Parkinson's volatility estimator using high-low range
                high_low_ratio = [np.log(bar['high'] / bar['low']) for bar in ohlc]
                if not high_low_ratio:
                    return 0.15
                
                return np.sqrt(sum(hl**2 for hl in high_low_ratio) / (4 * np.log(2) * len(high_low_ratio))) * np.sqrt(252)
            
            else:
                # Default to ATR calculation
                if len(ohlc) < 2:
                    return 0.15
                
                # Calculate simple ATR
                true_ranges = []
                for i in range(1, len(ohlc)):
                    high = ohlc[i]['high']
                    low = ohlc[i]['low']
                    prev_close = ohlc[i-1]['close']
                    
                    tr1 = high - low
                    tr2 = abs(high - prev_close)
                    tr3 = abs(low - prev_close)
                    
                    true_ranges.append(max(tr1, tr2, tr3))
                
                if not true_ranges:
                    return 0.15
                
                atr = sum(true_ranges) / len(true_ranges)
                price = ohlc[-1]['close']
                
                return atr / price
        
        # Default volatility if no data available
        return 0.15
    
    def get_strategy_allocation(
        self,
        strategy: str,
        account_value: float,
        market_regime: Optional[str] = None
    ) -> float:
        """
        Get the maximum allocation for a strategy based on performance and regime.
        
        Args:
            strategy: Strategy name
            account_value: Current account value
            market_regime: Optional market regime
            
        Returns:
            Maximum allocation percentage (0.0-1.0)
        """
        # Start with default allocation
        allocation = 0.3  # 30% default
        
        # Use stored metrics if available
        metrics = self.strategy_metrics.get(strategy)
        
        if metrics:
            trades = metrics.get('trades', 0)
            if trades >= 10:
                # Calculate Sharpe Ratio-like metric
                total_pnl = metrics.get('total_pnl', 0)
                pnl_series = [metrics.get('trade_pnls', [])]
                
                if pnl_series and len(pnl_series) > 1:
                    sharpe = np.mean(pnl_series) / (np.std(pnl_series) + 1e-9)
                    # Scale allocation by sharpe
                    if sharpe > 2:
                        allocation = 0.4  # 40% for high sharpe
                    elif sharpe > 1:
                        allocation = 0.35  # 35% for good sharpe
                    elif sharpe < 0.5:
                        allocation = 0.2  # 20% for poor sharpe
                    elif sharpe < 0:
                        allocation = 0.1  # 10% for negative sharpe
                
                # Win rate adjustment
                win_rate = metrics.get('wins', 0) / trades
                if win_rate > 0.7:
                    allocation *= 1.2  # Boost for high win rate
                elif win_rate < 0.4:
                    allocation *= 0.8  # Reduce for low win rate
                
                # Profitability adjustment
                profit_pct = total_pnl / account_value if account_value > 0 else 0
                if profit_pct > 0.1:  # More than 10% profit
                    allocation *= 1.15  # Slight boost for profitable strategies
                elif profit_pct < -0.05:  # More than 5% loss
                    allocation *= 0.7  # Significant reduction for losing strategies
        
        # Market regime adjustment
        if market_regime and market_regime in self.regime_adjustments:
            regime_factor = self.regime_adjustments[market_regime]
            allocation *= regime_factor
        
        # Cap at 50% maximum
        return min(0.5, allocation)
    
    def recalculate_all_adjustments(self):
        """Recalculate all adjustment factors based on current metrics"""
        # Clear the calculation cache
        self.calculation_cache = {}
        
        logger.info("Recalculated all position sizing adjustments")
