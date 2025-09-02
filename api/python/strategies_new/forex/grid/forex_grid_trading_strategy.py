"""
Forex Grid Trading Strategy

This strategy implements a dynamic grid trading approach for forex pairs, automatically
placing buy and sell orders at predetermined price levels to profit from price oscillations.
The grid spacing is dynamically adjusted based on volatility, and positions are managed
with sophisticated risk controls and profit-taking mechanisms.

Features:
- Dynamic grid spacing based on ATR and volatility analysis
- Adaptive grid rebalancing as price evolves
- Multiple grid modes: symmetric, trend-biased, and volatility-weighted
- Advanced position sizing and risk management
- Profit accumulation with partial position closing
- Grid boundary protection mechanisms
- Performance tracking and analytics
"""

import logging
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Set

from trading_bot.strategies_new.forex.base.forex_base_strategy import ForexBaseStrategy
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.models.signal import Signal


@register_strategy(
    asset_class="forex",
    strategy_type="grid",
    name="ForexGridTrading",
    description="Systematic grid trading strategy with dynamic spacing based on volatility and trend conditions",
    parameters={
        "default": {
            # Grid parameters
            "grid_type": "symmetric",  # Options: symmetric, trend_biased, volatility_weighted
            "num_grid_levels": 10,  # Number of grid levels in each direction
            "base_grid_spacing_pips": 20,  # Base spacing between grid levels in pips
            "max_grid_spacing_pips": 100,  # Maximum grid spacing in pips
            "min_grid_spacing_pips": 5,  # Minimum grid spacing in pips
            "spacing_atr_factor": 0.5,  # Multiply ATR by this factor for dynamic grid spacing
            "grid_activation_threshold": 0.5,  # Require this portion of grid levels to be set before trading
            
            # Volatility parameters
            "atr_period": 14,
            "atr_timeframe": "4h",
            "volatility_lookback_periods": 5,  # Number of periods to look back for volatility calculation
            
            # Trend parameters (for trend-biased grids)
            "trend_indicator": "ema",  # Options: ema, macd, adx
            "fast_ema_period": 20,
            "slow_ema_period": 50,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "adx_period": 14,
            "trend_threshold": 25,  # Minimum ADX value to consider a trend
            "trend_bias_multiplier": 1.5,  # Adjust grid spacing in trend direction
            
            # Position sizing and risk management
            "account_risk_per_grid": 0.005,  # Risk 0.5% of account per complete grid
            "max_account_risk": 0.05,  # Maximum 5% account risk across all grids
            "position_size_per_level": 0.1,  # Size at each level as a fraction of total grid size
            "max_levels_per_side": 5,  # Maximum active levels on buy or sell side
            "max_total_positions": 20,  # Maximum total positions across all pairs
            "max_position_size": 1.0,  # Maximum position size in standard lots
            
            # Profit-taking parameters
            "profit_target_pips": 15,  # Take profit for individual grid levels
            "complete_grid_profit_target_pct": 3.0,  # Take profit for the entire grid
            "partial_close_pct": 50,  # Close this percentage of position at first target
            "trailing_stop_activation_pips": 30,  # Activate trailing stop after this many pips in profit
            "trailing_stop_distance_pips": 15,  # Trailing stop distance
            
            # Grid management and rebalancing
            "grid_rebalance_frequency": "daily",  # Options: never, daily, weekly, volatility_change
            "grid_lifetime_days": 30,  # Maximum days to keep a grid active
            "max_drawdown_pct": 10,  # Maximum drawdown percentage before grid adjustment
            "min_profitability_threshold": -2.0,  # Minimum profitability before considering grid adjustment
            
            # Execution parameters
            "order_expiry_hours": 24,
            "order_refresh_frequency": 4,  # Hours between refreshing pending orders
            "boundary_buffer_pips": 10,  # Buffer pips from key levels
            
            # General parameters
            "timezone": pytz.UTC,
            "preferred_pairs": [
                "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
                "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "EURCHF"
            ],
            "pair_selection_count": 3  # Number of pairs to trade concurrently
        },
        
        # Configuration for highly volatile markets
        "high_volatility": {
            "grid_type": "volatility_weighted", 
            "base_grid_spacing_pips": 40,
            "max_grid_spacing_pips": 200,
            "min_grid_spacing_pips": 20,
            "spacing_atr_factor": 0.8,
            "max_levels_per_side": 3,
            "profit_target_pips": 30,
            "trailing_stop_distance_pips": 25,
            "grid_rebalance_frequency": "volatility_change"
        },
        
        # Configuration for trending markets
        "trending_market": {
            "grid_type": "trend_biased", 
            "trend_bias_multiplier": 2.0,
            "num_grid_levels": 12,
            "base_grid_spacing_pips": 30,
            "trend_threshold": 20,
            "grid_rebalance_frequency": "weekly",
            "partial_close_pct": 75  # Take more profit in trending markets
        }
    }
)
class ForexGridTradingStrategy(ForexBaseStrategy):
    """
    A strategy that implements grid trading for forex pairs.
    
    Grid trading involves placing buy and sell orders at predetermined levels (a grid),
    allowing the strategy to profit from price oscillations within a range or
    following a trend by systematically buying low and selling high.
    """
    
    def __init__(self, session=None):
        """
        Initialize the grid trading strategy.
        
        Args:
            session: Trading session object with configuration
        """
        super().__init__(session)
        self.name = "ForexGridTrading"
        self.description = "Dynamic grid trading with volatility-based spacing"
        self.logger = logging.getLogger(__name__)
        
        # Active grids and orders tracking
        self.active_grids = {}  # symbol -> grid data
        self.pending_orders = {}  # order_id -> order data
        self.active_positions = {}  # position_id -> position data
        
        # Performance tracking
        self.stats = {
            "grids_created": 0,
            "orders_placed": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "total_profit_pips": 0,
            "total_profit_amount": 0,
            "best_grid_profit_pct": 0,
            "worst_grid_loss_pct": 0,
            "avg_profit_per_grid": 0
        }
        
        # Last rebalance tracking
        self.last_rebalance_time = {}  # symbol -> last rebalance time
        
    def initialize(self) -> None:
        """Initialize strategy and load any required data."""
        super().initialize()
        
        # Initialize active grids
        self.active_grids = {}
        self.pending_orders = {}
        self.active_positions = {}
        
        self.logger.info(f"Initialized {self.name} strategy")
        
        # Select initial pairs for grid trading
        self._select_trading_pairs()
    
    def _select_trading_pairs(self) -> List[str]:
        """
        Select pairs for grid trading based on volatility, liquidity, and correlation.
        
        Returns:
            List of selected pairs for grid trading
        """
        # For now, we'll just use the top N pairs from the preferred pairs list
        # In a real implementation, this would analyze volatility, spreads, and correlations
        # to select the optimal set of pairs for grid trading
        
        preferred_pairs = self.parameters["preferred_pairs"]
        count = min(self.parameters["pair_selection_count"], len(preferred_pairs))
        
        selected_pairs = preferred_pairs[:count]
        
        self.logger.info(f"Selected {len(selected_pairs)} pairs for grid trading: {selected_pairs}")
        
        return selected_pairs
    
    def _calculate_volatility(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate volatility metrics for a price series.
        
        Args:
            data: OHLCV data
            
        Returns:
            Dictionary with volatility metrics
        """
        if data.empty or len(data) < self.parameters["atr_period"]:
            return {"atr": 0.0, "atr_pips": 0.0, "volatility_rank": 0.0}
        
        # Calculate ATR
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.parameters["atr_period"]).mean().iloc[-1]
        
        # Convert ATR to pips
        # Determine if this is a JPY pair (different pip value)
        is_jpy_pair = "JPY" in data.name if hasattr(data, "name") else False
        pip_value = 0.01 if is_jpy_pair else 0.0001
        atr_pips = atr / pip_value
        
        # Calculate volatility rank (percentile of current ATR compared to history)
        lookback = self.parameters["volatility_lookback_periods"]
        atr_history = true_range.rolling(window=self.parameters["atr_period"]).mean()
        if len(atr_history) > lookback:
            recent_atrs = atr_history.iloc[-lookback:]
            volatility_rank = (atr - recent_atrs.min()) / (recent_atrs.max() - recent_atrs.min()) \
                if recent_atrs.max() > recent_atrs.min() else 0.5
        else:
            volatility_rank = 0.5  # Default to mid-range if not enough history
        
        return {
            "atr": atr,
            "atr_pips": atr_pips,
            "volatility_rank": volatility_rank
        }
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trend direction and strength.
        
        Args:
            data: OHLCV data
            
        Returns:
            Dictionary with trend metrics
        """
        if data.empty or len(data) < 50:  # Need enough data for trend analysis
            return {"trend_direction": "neutral", "trend_strength": 0.0}
        
        close = data["close"]
        trend_indicator = self.parameters["trend_indicator"]
        
        # EMA crossover trend detection
        if trend_indicator == "ema":
            fast_period = self.parameters["fast_ema_period"]
            slow_period = self.parameters["slow_ema_period"]
            
            if len(close) <= slow_period:
                return {"trend_direction": "neutral", "trend_strength": 0.0}
                
            fast_ema = close.ewm(span=fast_period, adjust=False).mean()
            slow_ema = close.ewm(span=slow_period, adjust=False).mean()
            
            # Determine trend direction
            current_fast = fast_ema.iloc[-1]
            current_slow = slow_ema.iloc[-1]
            prev_fast = fast_ema.iloc[-2]
            prev_slow = slow_ema.iloc[-2]
            
            # Trend direction
            if current_fast > current_slow:
                trend_direction = "bullish"
                # Strength based on separation and slope
                trend_strength = min(1.0, (current_fast - current_slow) / current_slow * 100)
            elif current_fast < current_slow:
                trend_direction = "bearish"
                trend_strength = min(1.0, (current_slow - current_fast) / current_fast * 100)
            else:
                trend_direction = "neutral"
                trend_strength = 0.0
                
            # Adjust strength based on momentum (if emas are widening or narrowing)
            current_diff = abs(current_fast - current_slow)
            prev_diff = abs(prev_fast - prev_slow)
            
            if current_diff > prev_diff:
                # Trend is strengthening
                trend_strength *= 1.2
            else:
                # Trend is weakening
                trend_strength *= 0.8
                
        # MACD trend detection
        elif trend_indicator == "macd":
            fast_period = self.parameters["macd_fast"]
            slow_period = self.parameters["macd_slow"]
            signal_period = self.parameters["macd_signal"]
            
            if len(close) <= slow_period + signal_period:
                return {"trend_direction": "neutral", "trend_strength": 0.0}
                
            # Calculate MACD
            fast_ema = close.ewm(span=fast_period, adjust=False).mean()
            slow_ema = close.ewm(span=slow_period, adjust=False).mean()
            macd = fast_ema - slow_ema
            signal = macd.ewm(span=signal_period, adjust=False).mean()
            histogram = macd - signal
            
            # Determine trend direction and strength
            if macd.iloc[-1] > signal.iloc[-1]:
                trend_direction = "bullish"
                # Strength based on histogram value and whether it's growing
                trend_strength = min(1.0, abs(histogram.iloc[-1] / close.iloc[-1] * 1000))
                if histogram.iloc[-1] > histogram.iloc[-2]:
                    trend_strength *= 1.2  # Strengthening
            elif macd.iloc[-1] < signal.iloc[-1]:
                trend_direction = "bearish"
                trend_strength = min(1.0, abs(histogram.iloc[-1] / close.iloc[-1] * 1000))
                if histogram.iloc[-1] < histogram.iloc[-2]:
                    trend_strength *= 1.2  # Strengthening
            else:
                trend_direction = "neutral"
                trend_strength = 0.0
                
        # ADX trend detection
        elif trend_indicator == "adx":
            adx_period = self.parameters["adx_period"]
            
            if len(data) <= adx_period * 2:
                return {"trend_direction": "neutral", "trend_strength": 0.0}
                
            # Calculate ADX (simplified implementation)
            high = data["high"]
            low = data["low"]
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate +DM and -DM
            plus_dm = high.diff()
            minus_dm = low.diff().multiply(-1)
            
            # When +DM is larger and positive, keep +DM, otherwise set to 0
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
            
            # When -DM is larger and positive, keep -DM, otherwise set to 0
            minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
            
            # Calculate smoothed values
            tr_smooth = tr.rolling(window=adx_period).mean()
            plus_di = 100 * (plus_dm.rolling(window=adx_period).mean() / tr_smooth)
            minus_di = 100 * (minus_dm.rolling(window=adx_period).mean() / tr_smooth)
            
            # Calculate DX and ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
            adx = dx.rolling(window=adx_period).mean()
            
            # Get current values
            current_adx = adx.iloc[-1]
            current_plus_di = plus_di.iloc[-1]
            current_minus_di = minus_di.iloc[-1]
            
            # Determine trend direction and strength
            trend_threshold = self.parameters["trend_threshold"]
            if current_adx > trend_threshold:
                if current_plus_di > current_minus_di:
                    trend_direction = "bullish"
                else:
                    trend_direction = "bearish"
                # Normalize ADX as trend strength (typically 0-100, normalize to 0-1)
                trend_strength = min(1.0, current_adx / 100)
            else:
                trend_direction = "neutral"
                trend_strength = min(1.0, current_adx / trend_threshold)
                
        else:  # Default fallback
            trend_direction = "neutral"
            trend_strength = 0.0
            
        return {
            "trend_direction": trend_direction,
            "trend_strength": trend_strength
        }
    
    def _create_grid(self, symbol: str, data: pd.DataFrame, volatility: Dict[str, float], 
                     trend: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a trading grid for a currency pair.
        
        Args:
            symbol: Currency pair symbol
            data: OHLCV data
            volatility: Volatility metrics
            trend: Trend analysis results
            
        Returns:
            Dictionary with grid configuration
        """
        if data.empty:
            self.logger.warning(f"Cannot create grid for {symbol} with empty data")
            return None
        
        # Get current price
        current_price = data["close"].iloc[-1]
        
        # Determine grid type and spacing
        grid_type = self.parameters["grid_type"]
        base_spacing_pips = self.parameters["base_grid_spacing_pips"]
        min_spacing_pips = self.parameters["min_grid_spacing_pips"]
        max_spacing_pips = self.parameters["max_grid_spacing_pips"]
        num_levels = self.parameters["num_grid_levels"]
        atr_factor = self.parameters["spacing_atr_factor"]
        
        # Determine pip value
        pip_value = 0.01 if "JPY" in symbol else 0.0001
        
        # Adjust grid spacing based on volatility
        if volatility["atr_pips"] > 0:
            grid_spacing_pips = min(max_spacing_pips, 
                                  max(min_spacing_pips, 
                                    base_spacing_pips * (1 + volatility["volatility_rank"])))
        else:
            grid_spacing_pips = base_spacing_pips
        
        # Convert to price points
        grid_spacing = grid_spacing_pips * pip_value
        
        # For trend-biased grids, adjust spacing based on trend direction
        if grid_type == "trend_biased" and trend["trend_direction"] != "neutral":
            trend_bias = self.parameters["trend_bias_multiplier"]
            
            if trend["trend_direction"] == "bullish":
                # Wider spacing for sell orders (against trend), tighter for buy orders (with trend)
                upward_spacing = grid_spacing / trend_bias
                downward_spacing = grid_spacing * trend_bias
            else:  # bearish
                # Wider spacing for buy orders (against trend), tighter for sell orders (with trend)
                upward_spacing = grid_spacing * trend_bias
                downward_spacing = grid_spacing / trend_bias
        else:
            # Symmetric grid
            upward_spacing = grid_spacing
            downward_spacing = grid_spacing
        
        # Create grid levels
        buy_levels = []
        sell_levels = []
        
        # For buy orders (below current price)
        for i in range(1, num_levels + 1):
            price = current_price - (downward_spacing * i)
            level = {
                "price": price,
                "type": "buy",
                "level": i,
                "status": "pending",
                "position_size": None,  # Will be calculated at order placement time
                "order_id": None,
                "position_id": None,
                "created_time": datetime.now(self.parameters["timezone"])
            }
            buy_levels.append(level)
            
        # For sell orders (above current price)
        for i in range(1, num_levels + 1):
            price = current_price + (upward_spacing * i)
            level = {
                "price": price,
                "type": "sell",
                "level": i,
                "status": "pending",
                "position_size": None,
                "order_id": None,
                "position_id": None,
                "created_time": datetime.now(self.parameters["timezone"])
            }
            sell_levels.append(level)
            
        # Calculate grid boundaries
        lower_boundary = buy_levels[-1]["price"] - (downward_spacing / 2)
        upper_boundary = sell_levels[-1]["price"] + (upward_spacing / 2)
        
        # Create grid data structure
        grid = {
            "symbol": symbol,
            "created_time": datetime.now(self.parameters["timezone"]),
            "last_update_time": datetime.now(self.parameters["timezone"]),
            "current_price": current_price,
            "grid_type": grid_type,
            "buy_levels": buy_levels,
            "sell_levels": sell_levels,
            "lower_boundary": lower_boundary,
            "upper_boundary": upper_boundary,
            "grid_spacing_pips": grid_spacing_pips,
            "upward_spacing": upward_spacing,
            "downward_spacing": downward_spacing,
            "volatility": volatility,
            "trend": trend,
            "status": "active",
            "total_profit_pips": 0,
            "realized_profit": 0,
            "unrealized_profit": 0,
            "active_positions_count": 0,
            "completed_trades_count": 0,
            "orders_filled_count": 0,
            "next_rebalance_time": self._calculate_next_rebalance_time(symbol)
        }
        
        # Calculate profit target for the entire grid
        grid_profit_target = (self.parameters["complete_grid_profit_target_pct"] / 100) * \
                             (self.session.account_balance if self.session else 10000)
        grid["grid_profit_target"] = grid_profit_target
        
        # Update statistics
        self.stats["grids_created"] += 1
        
        self.logger.info(f"Created {grid_type} grid for {symbol} with {num_levels*2} levels, "
                       f"spacing: {grid_spacing_pips:.1f} pips, boundaries: "
                       f"{lower_boundary:.5f} - {upper_boundary:.5f}")
        
        return grid
    
    def _calculate_next_rebalance_time(self, symbol: str) -> datetime:
        """
        Calculate the next time to rebalance the grid.
        
        Args:
            symbol: Currency pair symbol
            
        Returns:
            Datetime of next scheduled rebalance
        """
        current_time = datetime.now(self.parameters["timezone"])
        rebalance_frequency = self.parameters["grid_rebalance_frequency"]
        
        if rebalance_frequency == "never":
            # Set to far future
            return current_time + timedelta(days=365)
        elif rebalance_frequency == "daily":
            return current_time + timedelta(days=1)
        elif rebalance_frequency == "weekly":
            return current_time + timedelta(days=7)
        else:  # volatility_change - handled dynamically
            return current_time + timedelta(days=3)  # Default fallback
    
    def _calculate_position_size(self, symbol: str, grid: Dict[str, Any], level: Dict[str, Any]) -> float:
        """
        Calculate position size for a grid level based on risk parameters.
        
        Args:
            symbol: Currency pair symbol
            grid: Grid configuration
            level: The specific grid level
            
        Returns:
            Position size in lots
        """
        # Get account balance and risk parameters
        account_balance = self.session.account_balance if self.session else 10000
        risk_per_grid = self.parameters["account_risk_per_grid"]
        max_account_risk = self.parameters["max_account_risk"]
        position_size_per_level = self.parameters["position_size_per_level"]
        max_position_size = self.parameters["max_position_size"]
        
        # Calculate total risk amount for this grid
        grid_risk_amount = account_balance * risk_per_grid
        
        # Account for current exposure
        total_active_grids = len(self.active_grids)
        if total_active_grids > 1:
            # Reduce risk if we have multiple active grids
            grid_risk_amount = min(grid_risk_amount, 
                                 account_balance * max_account_risk / total_active_grids)
        
        # Calculate position size for this level
        level_risk = grid_risk_amount * position_size_per_level
        
        # Scale based on level (deeper levels get smaller positions)
        level_scaling = max(0.3, 1.0 / level["level"])
        level_risk *= level_scaling
        
        # Determine pip value
        pip_value = 0.01 if "JPY" in symbol else 0.0001
        
        # For simple position sizing, use a fixed risk per pip
        # In a real implementation, this would include sophisticated risk calculations
        # including account currency conversion, leverage, etc.
        standard_lot_size = 100000  # 1 standard lot
        pip_risk = 5  # Risk 5 pips per level
        
        # Calculate position size in standard lots
        position_size = level_risk / (pip_risk * pip_value * standard_lot_size)
        
        # Ensure within limits
        position_size = min(max_position_size, position_size)
        
        return position_size
    
    def _prepare_grid_orders(self, symbol: str, grid: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare orders for a grid's levels that don't have active orders.
        
        Args:
            symbol: Currency pair symbol
            grid: Grid configuration
            
        Returns:
            List of orders to be placed
        """
        orders = []
        current_time = datetime.now(self.parameters["timezone"])
        
        # Check if we've exceeded max positions
        active_positions_count = sum(1 for pos in self.active_positions.values() 
                                   if pos["symbol"] == symbol)
        max_levels_per_side = self.parameters["max_levels_per_side"]
        
        # Process buy levels (from lowest level to highest, as lower levels have priority)
        buy_levels = sorted(grid["buy_levels"], key=lambda x: x["level"], reverse=True)
        sell_levels = sorted(grid["sell_levels"], key=lambda x: x["level"])
        
        # Count active buy and sell positions
        active_buys = sum(1 for pos in self.active_positions.values() 
                         if pos["symbol"] == symbol and pos["side"] == "buy")
        active_sells = sum(1 for pos in self.active_positions.values() 
                          if pos["symbol"] == symbol and pos["side"] == "sell")
        
        # Process buy levels
        for level in buy_levels:
            # Skip if already has an order or position
            if level["status"] != "pending":
                continue
                
            # Skip if we've reached max positions for this side
            if active_buys >= max_levels_per_side:
                continue
                
            # Calculate position size for this level
            position_size = self._calculate_position_size(symbol, grid, level)
            
            # Create order
            order = {
                "symbol": symbol,
                "side": "buy",
                "order_type": "limit",
                "price": level["price"],
                "position_size": position_size,
                "level": level["level"],
                "grid_id": id(grid),  # Use object ID as unique identifier
                "time_created": current_time,
                "expiry_time": current_time + timedelta(hours=self.parameters["order_expiry_hours"])
            }
            
            orders.append(order)
            level["position_size"] = position_size
            level["status"] = "order_placed"
            active_buys += 1
            
            # Stop if we've reached max positions for this side
            if active_buys >= max_levels_per_side:
                break
        
        # Process sell levels
        for level in sell_levels:
            # Skip if already has an order or position
            if level["status"] != "pending":
                continue
                
            # Skip if we've reached max positions for this side
            if active_sells >= max_levels_per_side:
                continue
                
            # Calculate position size for this level
            position_size = self._calculate_position_size(symbol, grid, level)
            
            # Create order
            order = {
                "symbol": symbol,
                "side": "sell",
                "order_type": "limit",
                "price": level["price"],
                "position_size": position_size,
                "level": level["level"],
                "grid_id": id(grid),  # Use object ID as unique identifier
                "time_created": current_time,
                "expiry_time": current_time + timedelta(hours=self.parameters["order_expiry_hours"])
            }
            
            orders.append(order)
            level["position_size"] = position_size
            level["status"] = "order_placed"
            active_sells += 1
            
            # Stop if we've reached max positions for this side
            if active_sells >= max_levels_per_side:
                break
        
        # Update statistics
        self.stats["orders_placed"] += len(orders)
        
        return orders
    
    def _check_grid_rebalance(self, symbol: str, grid: Dict[str, Any], data: pd.DataFrame) -> bool:
        """
        Check if a grid needs to be rebalanced.
        
        Args:
            symbol: Currency pair symbol
            grid: Grid configuration
            data: OHLCV data
            
        Returns:
            True if grid should be rebalanced, False otherwise
        """
        current_time = datetime.now(self.parameters["timezone"])
        
        # Skip if grid is not active
        if grid["status"] != "active":
            return False
            
        # Skip if no data
        if data.empty:
            return False
            
        # Check scheduled rebalance time
        if current_time >= grid["next_rebalance_time"]:
            self.logger.info(f"Scheduled rebalance time reached for {symbol} grid")
            return True
            
        # Get current price
        current_price = data["close"].iloc[-1]
        
        # Check if price has moved outside grid boundaries
        if current_price <= grid["lower_boundary"] or current_price >= grid["upper_boundary"]:
            self.logger.info(f"Price {current_price:.5f} has moved outside grid boundaries for {symbol}")
            return True
            
        # Check for significant volatility change if using volatility-based rebalancing
        if self.parameters["grid_rebalance_frequency"] == "volatility_change":
            volatility = self._calculate_volatility(data)
            current_vol_rank = volatility["volatility_rank"]
            previous_vol_rank = grid["volatility"]["volatility_rank"]
            
            # Check if volatility rank has changed significantly
            if abs(current_vol_rank - previous_vol_rank) > 0.3:  # 30% change in volatility rank
                self.logger.info(f"Significant volatility change detected for {symbol} "
                               f"({previous_vol_rank:.2f} -> {current_vol_rank:.2f})")
                return True
        
        # Check if grid has been active for too long
        grid_lifetime_days = self.parameters["grid_lifetime_days"]
        if (current_time - grid["created_time"]).days > grid_lifetime_days:
            self.logger.info(f"Grid for {symbol} has exceeded maximum lifetime of {grid_lifetime_days} days")
            return True
            
        # Check grid performance
        if grid["active_positions_count"] > 0 and grid["realized_profit"] < 0:
            # Check if grid is underperforming
            min_profitability = self.parameters["min_profitability_threshold"]
            account_balance = self.session.account_balance if self.session else 10000
            profitability_pct = (grid["realized_profit"] / account_balance) * 100
            
            if profitability_pct < min_profitability:
                self.logger.info(f"Grid for {symbol} is underperforming: {profitability_pct:.2f}% profit")
                return True
        
        return False
    
    def _rebalance_grid(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Rebalance a grid for a currency pair.
        
        Args:
            symbol: Currency pair symbol
            data: OHLCV data
            
        Returns:
            New grid configuration
        """
        # Log rebalancing
        self.logger.info(f"Rebalancing grid for {symbol}")
        
        # Calculate new volatility and trend
        volatility = self._calculate_volatility(data)
        trend = self._analyze_trend(data)
        
        # Create new grid
        new_grid = self._create_grid(symbol, data, volatility, trend)
        
        # Track last rebalance time
        self.last_rebalance_time[symbol] = datetime.now(self.parameters["timezone"])
        
        return new_grid
    
    def on_data(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Process new market data and update grid strategy state.
        
        Args:
            data_dict: Dictionary of market data for different pairs
        """
        # Update existing grids first
        self._update_active_grids(data_dict)
        
        # Create new grids for pairs that don't have active grids
        self._create_new_grids(data_dict)
        
        # Process signals and emit them
        self._process_signals(data_dict)
    
    def _update_active_grids(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Update status of active grids.
        
        Args:
            data_dict: Dictionary of market data for different pairs
        """
        symbols_to_rebalance = []
        
        for symbol, grid in self.active_grids.items():
            # Skip if we don't have data for this symbol
            if symbol not in data_dict or data_dict[symbol].empty:
                continue
                
            # Check if grid needs rebalancing
            if self._check_grid_rebalance(symbol, grid, data_dict[symbol]):
                symbols_to_rebalance.append(symbol)
                continue
                
            # Update grid with current price
            current_price = data_dict[symbol]["close"].iloc[-1]
            grid["current_price"] = current_price
            grid["last_update_time"] = datetime.now(self.parameters["timezone"])
            
            # Prepare new orders if needed
            new_orders = self._prepare_grid_orders(symbol, grid)
            
            # In a real implementation, these orders would be sent to the broker
            # For now, we'll just log them and track in our local state
            for order in new_orders:
                order_id = f"order_{len(self.pending_orders) + 1}"
                self.pending_orders[order_id] = order
                
                # Update level with order ID
                level_type = "buy_levels" if order["side"] == "buy" else "sell_levels"
                level_index = order["level"] - 1  # Convert to 0-based index
                
                if 0 <= level_index < len(grid[level_type]):
                    grid[level_type][level_index]["order_id"] = order_id
                    
                self.logger.info(f"Placed {order['side']} order for {symbol} at {order['price']:.5f}, "
                               f"size: {order['position_size']:.2f} lots")
        
        # Rebalance grids that need it
        for symbol in symbols_to_rebalance:
            if symbol in data_dict and not data_dict[symbol].empty:
                # Cancel all pending orders for this grid
                self._cancel_grid_orders(symbol)
                
                # Create new grid
                new_grid = self._rebalance_grid(symbol, data_dict[symbol])
                
                # Replace old grid with new one
                self.active_grids[symbol] = new_grid
    
    def _cancel_grid_orders(self, symbol: str) -> None:
        """
        Cancel all pending orders for a grid.
        
        Args:
            symbol: Currency pair symbol
        """
        orders_to_cancel = []
        
        # Find orders for this symbol
        for order_id, order in self.pending_orders.items():
            if order["symbol"] == symbol:
                orders_to_cancel.append(order_id)
                
        # Cancel orders
        for order_id in orders_to_cancel:
            # In a real implementation, this would send a cancel request to the broker
            self.logger.info(f"Cancelling order {order_id} for {symbol}")
            self.pending_orders.pop(order_id, None)
            self.stats["orders_cancelled"] += 1
    
    def _create_new_grids(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Create new grids for pairs that don't have active grids.
        
        Args:
            data_dict: Dictionary of market data for different pairs
        """
        # Get pairs for potential grid trading
        tradable_pairs = self._select_trading_pairs()
        
        # Skip if we have reached the maximum number of concurrent grids
        if len(self.active_grids) >= self.parameters["pair_selection_count"]:
            return
            
        # Create grids for pairs that don't have active grids
        for symbol in tradable_pairs:
            # Skip if already have an active grid for this pair
            if symbol in self.active_grids:
                continue
                
            # Skip if we don't have data for this symbol
            if symbol not in data_dict or data_dict[symbol].empty:
                continue
                
            # Calculate volatility and trend
            volatility = self._calculate_volatility(data_dict[symbol])
            trend = self._analyze_trend(data_dict[symbol])
            
            # Create grid
            grid = self._create_grid(symbol, data_dict[symbol], volatility, trend)
            
            if grid:
                # Store grid
                self.active_grids[symbol] = grid
                
                # Prepare initial orders
                self._prepare_grid_orders(symbol, grid)
                
                # Break if we've reached the maximum number of concurrent grids
                if len(self.active_grids) >= self.parameters["pair_selection_count"]:
                    break
    
    def _process_signals(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Process signals for active grids.
        
        Args:
            data_dict: Dictionary of market data for different pairs
        """
        # In a real implementation, this would check for filled orders from the broker
        # and generate trade signals accordingly
        
        # For now, we'll simulate filled orders based on price movements
        signals = self._generate_signals(data_dict)
        
        # Emit signals
        for signal in signals:
            self.emit_signal(signal)
    
    def _generate_signals(self, data_dict: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate signals for active grids based on price movements.
        
        Args:
            data_dict: Dictionary of market data for different pairs
            
        Returns:
            List of signals
        """
        signals = []
        current_time = datetime.now(self.parameters["timezone"])
        
        # Check each pending order to see if it would have been filled
        orders_to_remove = []
        new_positions = []
        
        for order_id, order in self.pending_orders.items():
            symbol = order["symbol"]
            
            # Skip if we don't have data for this symbol
            if symbol not in data_dict or data_dict[symbol].empty:
                continue
                
            # Check if order has expired
            if current_time > order["expiry_time"]:
                orders_to_remove.append(order_id)
                self.logger.info(f"Order {order_id} for {symbol} has expired")
                continue
                
            # Get price data
            data = data_dict[symbol]
            current_price = data["close"].iloc[-1]
            low_price = data["low"].iloc[-1]
            high_price = data["high"].iloc[-1]
            
            # Check if order would have been filled
            order_filled = False
            fill_price = order["price"]
            
            if order["side"] == "buy" and low_price <= order["price"]:
                order_filled = True
            elif order["side"] == "sell" and high_price >= order["price"]:
                order_filled = True
                
            if order_filled:
                # Create a position from the filled order
                position_id = f"position_{len(self.active_positions) + 1}"
                position = {
                    "position_id": position_id,
                    "symbol": symbol,
                    "side": order["side"],
                    "entry_price": fill_price,
                    "position_size": order["position_size"],
                    "level": order["level"],
                    "grid_id": order["grid_id"],
                    "time_opened": current_time,
                    "take_profit": self._calculate_take_profit(symbol, order),
                    "trailing_stop": None,  # Will be activated later if price moves favorably
                    "status": "open"
                }
                
                # Store position
                new_positions.append((position_id, position))
                
                # Mark order for removal
                orders_to_remove.append(order_id)
                
                # Update grid and level information
                if symbol in self.active_grids:
                    grid = self.active_grids[symbol]
                    grid["orders_filled_count"] += 1
                    grid["active_positions_count"] += 1
                    
                    level_type = "buy_levels" if order["side"] == "buy" else "sell_levels"
                    level_index = order["level"] - 1  # Convert to 0-based index
                    
                    if 0 <= level_index < len(grid[level_type]):
                        grid[level_type][level_index]["status"] = "position_open"
                        grid[level_type][level_index]["position_id"] = position_id
                
                # Generate signal
                signal = Signal(
                    symbol=symbol,
                    signal_type=order["side"],
                    entry_price=fill_price,
                    stop_loss=None,  # Grid trading typically doesn't use traditional stop losses
                    take_profit=position["take_profit"],
                    size=order["position_size"],
                    timestamp=current_time,
                    timeframe=self.parameters["atr_timeframe"],
                    strategy=self.name,
                    strength=0.8,  # Grid signals are considered strong
                    metadata={
                        "grid_level": order["level"],
                        "grid_type": self.active_grids[symbol]["grid_type"] if symbol in self.active_grids else "unknown",
                        "position_id": position_id
                    }
                )
                
                signals.append(signal)
                self.stats["orders_filled"] += 1
                
                self.logger.info(f"Generated {order['side']} signal for {symbol} at {fill_price:.5f}, "
                              f"take profit: {position['take_profit']:.5f}")
        
        # Remove filled or expired orders
        for order_id in orders_to_remove:
            self.pending_orders.pop(order_id, None)
            
        # Add new positions
        for position_id, position in new_positions:
            self.active_positions[position_id] = position
            
        # Check if any positions need to be closed
        self._check_position_exits(data_dict, signals)
        
        return signals
    
    def _calculate_take_profit(self, symbol: str, order: Dict[str, Any]) -> float:
        """
        Calculate take profit level for a grid position.
        
        Args:
            symbol: Currency pair symbol
            order: Order information
            
        Returns:
            Take profit price
        """
        # Grid trading take profits are typically set at the next grid level in the opposite direction
        grid_id = order["grid_id"]
        level = order["level"]
        side = order["side"]
        price = order["price"]
        
        # Get grid information
        grid = next((g for g in self.active_grids.values() if id(g) == grid_id), None)
        
        if not grid:
            # If grid not found, use a fixed pip target
            pip_value = 0.01 if "JPY" in symbol else 0.0001
            profit_target_pips = self.parameters["profit_target_pips"]
            
            if side == "buy":
                return price + (profit_target_pips * pip_value)
            else:  # side == "sell"
                return price - (profit_target_pips * pip_value)
        
        # For grid trading, take profit is typically at the next grid level in the opposite direction
        if side == "buy":
            # For buy orders, take profit is at the next sell level above
            # Find the sell level just above this price
            target_level = None
            for sell_level in grid["sell_levels"]:
                if sell_level["price"] > price:
                    if target_level is None or sell_level["price"] < target_level["price"]:
                        target_level = sell_level
            
            if target_level:
                return target_level["price"]
            else:
                # Fallback: use fixed pip target
                pip_value = 0.01 if "JPY" in symbol else 0.0001
                profit_target_pips = self.parameters["profit_target_pips"]
                return price + (profit_target_pips * pip_value)
                
        else:  # side == "sell"
            # For sell orders, take profit is at the next buy level below
            # Find the buy level just below this price
            target_level = None
            for buy_level in grid["buy_levels"]:
                if buy_level["price"] < price:
                    if target_level is None or buy_level["price"] > target_level["price"]:
                        target_level = buy_level
            
            if target_level:
                return target_level["price"]
            else:
                # Fallback: use fixed pip target
                pip_value = 0.01 if "JPY" in symbol else 0.0001
                profit_target_pips = self.parameters["profit_target_pips"]
                return price - (profit_target_pips * pip_value)
    
    def _check_position_exits(self, data_dict: Dict[str, pd.DataFrame], signals: List[Signal]) -> None:
        """
        Check if any positions need to be closed.
        
        Args:
            data_dict: Dictionary of market data for different pairs
            signals: List of signals to append to
        """
        positions_to_close = []
        current_time = datetime.now(self.parameters["timezone"])
        
        for position_id, position in self.active_positions.items():
            symbol = position["symbol"]
            
            # Skip if we don't have data for this symbol
            if symbol not in data_dict or data_dict[symbol].empty:
                continue
                
            # Get price data
            data = data_dict[symbol]
            current_price = data["close"].iloc[-1]
            high_price = data["high"].iloc[-1]
            low_price = data["low"].iloc[-1]
            
            # Check if position should be closed
            close_position = False
            close_reason = ""
            realized_pips = 0
            
            # Take profit check
            if position["side"] == "buy":
                if high_price >= position["take_profit"]:
                    close_position = True
                    close_reason = "take_profit"
                    exit_price = position["take_profit"]
                    
                    # Calculate realized profit in pips
                    pip_value = 0.01 if "JPY" in symbol else 0.0001
                    realized_pips = (exit_price - position["entry_price"]) / pip_value
                    
            else:  # position["side"] == "sell"
                if low_price <= position["take_profit"]:
                    close_position = True
                    close_reason = "take_profit"
                    exit_price = position["take_profit"]
                    
                    # Calculate realized profit in pips
                    pip_value = 0.01 if "JPY" in symbol else 0.0001
                    realized_pips = (position["entry_price"] - exit_price) / pip_value
            
            # Trailing stop check
            if not close_position and position["trailing_stop"] is not None:
                trailing_stop = position["trailing_stop"]
                
                if position["side"] == "buy" and low_price <= trailing_stop:
                    close_position = True
                    close_reason = "trailing_stop"
                    exit_price = trailing_stop
                    
                    # Calculate realized profit in pips
                    pip_value = 0.01 if "JPY" in symbol else 0.0001
                    realized_pips = (exit_price - position["entry_price"]) / pip_value
                    
                elif position["side"] == "sell" and high_price >= trailing_stop:
                    close_position = True
                    close_reason = "trailing_stop"
                    exit_price = trailing_stop
                    
                    # Calculate realized profit in pips
                    pip_value = 0.01 if "JPY" in symbol else 0.0001
                    realized_pips = (position["entry_price"] - exit_price) / pip_value
            
            # Check for trailing stop activation or adjustment
            if not close_position and position["trailing_stop"] is None:
                # Check if price has moved favorably enough to activate trailing stop
                trailing_activation_pips = self.parameters["trailing_stop_activation_pips"]
                trailing_stop_distance_pips = self.parameters["trailing_stop_distance_pips"]
                pip_value = 0.01 if "JPY" in symbol else 0.0001
                
                if position["side"] == "buy":
                    price_movement = current_price - position["entry_price"]
                    price_movement_pips = price_movement / pip_value
                    
                    if price_movement_pips >= trailing_activation_pips:
                        # Activate trailing stop
                        trailing_stop = current_price - (trailing_stop_distance_pips * pip_value)
                        position["trailing_stop"] = trailing_stop
                        
                        self.logger.info(f"Activated trailing stop for {symbol} buy position at "
                                      f"{trailing_stop:.5f} (price: {current_price:.5f})")
                        
                elif position["side"] == "sell":
                    price_movement = position["entry_price"] - current_price
                    price_movement_pips = price_movement / pip_value
                    
                    if price_movement_pips >= trailing_activation_pips:
                        # Activate trailing stop
                        trailing_stop = current_price + (trailing_stop_distance_pips * pip_value)
                        position["trailing_stop"] = trailing_stop
                        
                        self.logger.info(f"Activated trailing stop for {symbol} sell position at "
                                      f"{trailing_stop:.5f} (price: {current_price:.5f})")
            
            # Check for trailing stop adjustment
            elif not close_position and position["trailing_stop"] is not None:
                trailing_stop_distance_pips = self.parameters["trailing_stop_distance_pips"]
                pip_value = 0.01 if "JPY" in symbol else 0.0001
                
                if position["side"] == "buy":
                    # For buy positions, trailing stop can only move up
                    new_stop = current_price - (trailing_stop_distance_pips * pip_value)
                    if new_stop > position["trailing_stop"]:
                        position["trailing_stop"] = new_stop
                        
                        self.logger.debug(f"Adjusted trailing stop for {symbol} buy position to {new_stop:.5f}")
                        
                elif position["side"] == "sell":
                    # For sell positions, trailing stop can only move down
                    new_stop = current_price + (trailing_stop_distance_pips * pip_value)
                    if new_stop < position["trailing_stop"]:
                        position["trailing_stop"] = new_stop
                        
                        self.logger.debug(f"Adjusted trailing stop for {symbol} sell position to {new_stop:.5f}")
            
            # If position should be closed, mark it and generate exit signal
            if close_position:
                positions_to_close.append((position_id, close_reason, exit_price, realized_pips))
                
                # Generate exit signal
                signal = Signal(
                    symbol=symbol,
                    signal_type="exit",
                    entry_price=None,
                    stop_loss=None,
                    take_profit=None,
                    size=position["position_size"],
                    timestamp=current_time,
                    timeframe=self.parameters["atr_timeframe"],
                    strategy=self.name,
                    strength=1.0,  # Exit signals are always strong
                    metadata={
                        "position_id": position_id,
                        "exit_reason": close_reason,
                        "exit_price": exit_price,
                        "profit_pips": realized_pips,
                        "grid_level": position["level"]
                    }
                )
                
                signals.append(signal)
                
                # Log the exit
                self.logger.info(f"Generated exit signal for {symbol} {position['side']} position at "
                               f"{exit_price:.5f}, reason: {close_reason}, profit: {realized_pips:.1f} pips")
        
        # Process closed positions
        for position_id, close_reason, exit_price, realized_pips in positions_to_close:
            position = self.active_positions[position_id]
            symbol = position["symbol"]
            
            # Update grid stats if grid still exists
            if symbol in self.active_grids:
                grid = self.active_grids[symbol]
                grid["active_positions_count"] -= 1
                grid["completed_trades_count"] += 1
                grid["total_profit_pips"] += realized_pips
                
                # Calculate profit amount
                pip_value_usd = 10  # Approximate USD value per pip for a standard lot (simplified)
                realized_profit = realized_pips * pip_value_usd * position["position_size"]
                grid["realized_profit"] += realized_profit
                self.stats["total_profit_amount"] += realized_profit
                
                # Update level status
                level_type = "buy_levels" if position["side"] == "buy" else "sell_levels"
                level_index = position["level"] - 1  # Convert to 0-based index
                
                if 0 <= level_index < len(grid[level_type]):
                    grid[level_type][level_index]["status"] = "pending"  # Reset for reuse
                    grid[level_type][level_index]["position_id"] = None
                    grid[level_type][level_index]["order_id"] = None
                
                # Update best/worst stats
                profit_pct = (realized_profit / self.session.account_balance) * 100 if self.session else 0
                
                if profit_pct > 0 and profit_pct > self.stats["best_grid_profit_pct"]:
                    self.stats["best_grid_profit_pct"] = profit_pct
                elif profit_pct < 0 and profit_pct < self.stats["worst_grid_loss_pct"]:
                    self.stats["worst_grid_loss_pct"] = profit_pct
            
            # Update strategy stats
            self.stats["total_profit_pips"] += realized_pips
            
            # Remove closed position
            self.active_positions.pop(position_id, None)
    
    def update(self) -> Dict[str, Any]:
        """
        Update strategy state and return current performance metrics.
        
        Returns:
            Dictionary with strategy performance and status information
        """
        # Calculate total positions and orders
        total_positions = len(self.active_positions)
        total_pending_orders = len(self.pending_orders)
        
        # Calculate average profit per grid
        completed_grids = self.stats["grids_created"] - len(self.active_grids)
        avg_profit_per_grid = 0
        if completed_grids > 0:
            avg_profit_per_grid = self.stats["total_profit_amount"] / completed_grids
        
        # Return current status and performance metrics
        return {
            "strategy_name": self.name,
            "active_grids": len(self.active_grids),
            "active_positions": total_positions,
            "pending_orders": total_pending_orders,
            "active_pairs": list(self.active_grids.keys()),
            "orders_placed": self.stats["orders_placed"],
            "orders_filled": self.stats["orders_filled"],
            "total_profit_pips": self.stats["total_profit_pips"],
            "total_profit_amount": self.stats["total_profit_amount"],
            "best_grid_profit_pct": self.stats["best_grid_profit_pct"],
            "worst_grid_loss_pct": self.stats["worst_grid_loss_pct"],
            "avg_profit_per_grid": avg_profit_per_grid,
            "last_update": datetime.now(self.parameters["timezone"]).isoformat()
        }
    
    def shutdown(self) -> None:
        """
        Clean up resources and prepare for shutdown.
        """
        self.logger.info(f"Shutting down {self.name} strategy")
        self.logger.info(f"Strategy stats: {self.stats}")
        
        # Log details about active grids
        self.logger.info(f"Active grids at shutdown: {len(self.active_grids)}")
        self.logger.info(f"Active positions at shutdown: {len(self.active_positions)}")
        
        # Clear internal state
        self.active_grids.clear()
        self.pending_orders.clear()
        self.active_positions.clear()
        
        super().shutdown()
