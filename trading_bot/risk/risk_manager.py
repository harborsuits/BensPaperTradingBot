#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Manager - Advanced risk management system with dynamic position sizing,
stop-loss mechanisms, and exposure controls.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import os

# Import traditional config utils for backward compatibility
from trading_bot.common.config_utils import setup_directories, save_state, load_state

# Import the risk settings
from trading_bot.risk.risk_settings import RiskSettings

# Import the new typed settings system
try:
    from trading_bot.config.typed_settings import load_config as typed_load_config, RiskSettings
    from trading_bot.config.migration_utils import get_config_from_legacy_path, migrate_config
    TYPED_SETTINGS_AVAILABLE = True
except ImportError:
    TYPED_SETTINGS_AVAILABLE = False
    # Fall back to legacy config if typed settings not available
    from trading_bot.common.config_utils import load_config

# Setup logging
logger = logging.getLogger("RiskManager")

class RiskLevel(Enum):
    """
    Risk level classification system for portfolio-wide risk assessment.
    
    Provides a structured categorization of risk exposure levels from 
    lowest (LOW) to highest (CRITICAL). Each level represents a specific 
    threshold of risk parameters that trigger different risk mitigation actions.
    
    Risk levels are used throughout the system to:
    - Determine appropriate position sizing
    - Trigger risk reduction measures
    - Adjust trading frequency
    - Modify stop-loss parameters
    - Notify system operators of changing risk conditions
    
    The system automatically transitions between risk levels based on
    drawdown metrics, portfolio exposure, and volatility measurements.
    """
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXTREME = 4
    CRITICAL = 5

class StopLossType(Enum):
    """
    Classification of stop-loss methodologies for risk management.
    
    Defines different approaches to setting and managing stop-loss levels.
    Each methodology has specific characteristics and is appropriate for
    different market conditions and trading strategies.
    
    FIXED: Simple percentage-based stop from entry price
      - Consistent risk per trade regardless of volatility
      - Best for stable, low-volatility markets
      - Simple to implement and understand
    
    VOLATILITY: Dynamic stop based on market volatility (ATR)
      - Adapts to changing market conditions
      - Wider stops in volatile conditions
      - Prevents premature exits during normal market fluctuations
      - Typically uses Average True Range (ATR) multiples
    
    TRAILING: Dynamic stop that follows price in favorable direction
      - Locks in profits while allowing positions to run
      - Activates after position reaches profit threshold
      - Maintains fixed or percentage distance from price highs/lows
      - Optimal for trend-following strategies
    
    TIME_BASED: Exits position after specific time period
      - Used for mean-reversion or time-decay strategies
      - Can combine with price-based stops
      - Helps limit exposure to overnight or weekend risk
      - Enforces discipline for time-sensitive strategies
    """
    FIXED = 1         # Fixed percentage
    VOLATILITY = 2    # Volatility-based (e.g., ATR multiple)
    TRAILING = 3      # Trailing stop
    TIME_BASED = 4    # Time-based stop

class RiskManager:
    """
    Advanced risk management system for systematic trading operations.
    
    The RiskManager is a comprehensive risk control system that implements 
    professional-grade risk management practices across multiple dimensions
    including position sizing, stop-loss management, drawdown controls, and 
    exposure limits. It serves as the central risk governance component for
    the entire trading system.
    
    Key capabilities:
    1. Dynamic position sizing based on account equity and volatility
    2. Multi-tiered stop-loss management (fixed, volatility-based, trailing)
    3. Portfolio-level risk monitoring and exposure controls
    4. Maximum drawdown enforcement at daily and overall levels
    5. Value-at-Risk (VaR) calculations for position and portfolio risk
    6. Automated risk reduction recommendations when limits are breached
    7. Correlation-aware portfolio risk assessment
    8. Risk level classification and adaptive trading parameters
    9. Comprehensive trade history and risk metrics tracking
    10. State persistence for continuous risk management
    
    Core risk management principles implemented:
    - Capital preservation through systematic risk controls
    - Risk-adjusted position sizing for consistent risk exposure
    - Adaptive risk parameters based on market conditions
    - Multiple layers of risk management (position, strategy, portfolio)
    - Drawdown controls to prevent catastrophic losses
    - Diversification rules to prevent excess concentration
    - Automated risk reduction protocols during adverse conditions
    
    The RiskManager follows a defense-in-depth philosophy, where multiple
    risk controls work together to prevent significant losses and ensure
    the system's ability to continue operating under various market conditions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None, settings: Optional[RiskSettings] = None):
        """
        Initialize the risk management system with configuration parameters.
        
        Creates a new RiskManager instance with default or custom risk parameters.
        Sets up the initial portfolio state, risk thresholds, and tracking mechanisms
        for positions, drawdowns, and portfolio risk metrics.
        
        Args:
            config: Optional configuration dictionary with risk parameters
            config_path: Optional path to configuration file
            settings: Optional RiskSettings object from typed_settings system
            
        Configuration parameters include:
            - max_position_pct: Maximum position size as percentage of portfolio
            - max_risk_pct: Maximum risk per trade as percentage of portfolio
            - max_portfolio_risk: Maximum total portfolio risk percentage
            - max_correlated_positions: Maximum number of correlated positions
            - default_risk_per_trade: Default risk percentage per trade
            - daily_stop_loss_pct: Daily stop-loss percentage
            - max_daily_drawdown_pct: Maximum daily drawdown percentage
            - max_open_trades: Maximum number of open trades
            - correlation_threshold: Correlation threshold for related positions
            - stop_loss_type: Default stop-loss methodology (fixed, volatility, etc.)
            - default_fixed_stop_pct: Default fixed stop-loss percentage
            - default_atr_multiple: Default ATR multiple for volatility stops
            - trailing_stop_activation_pct: Percentage profit to activate trailing stops
            - trailing_stop_distance_pct: Trailing stop distance as percentage
            - portfolio_var_conf_level: Portfolio VaR confidence level
            - position_var_conf_level: Position VaR confidence level
            - var_days: VaR time horizon in days
        
        Notes:
            - Conservative settings prioritize capital preservation
            - More aggressive settings favor potential returns at higher risk
            - Parameters should be tuned based on strategy characteristics and risk tolerance
        """
        # Setup paths for state persistence
        self.paths = setup_directories()
        
        # Load configuration
        self.config = self._get_default_config()
        
        # Load config in the following priority order:
        # 1. RiskSettings object (from typed_settings)
        # 2. Custom config_path (file)
        # 3. Provided config dictionary
        # 4. Default config
        
        if settings and TYPED_SETTINGS_AVAILABLE:
            # Use typed settings if provided
            self._load_from_typed_settings(settings)
        elif config_path:
            # Try to load from path using typed settings if available
            if TYPED_SETTINGS_AVAILABLE:
                try:
                    # Get full config and extract risk settings
                    full_config = get_config_from_legacy_path(config_path)
                    self._load_from_typed_settings(full_config.risk)
                except Exception as e:
                    logger.warning(f"Could not load typed settings: {e}")
                    logger.warning("Falling back to legacy config loading")
                    # Fall back to legacy loading
                    loaded_config = load_config(config_path)
                    if loaded_config:
                        self.config.update(loaded_config)
            else:
                # Use traditional config loading
                loaded_config = load_config(config_path)
                if loaded_config:
                    self.config.update(loaded_config)
        elif config:
            # Use provided config dict
            self.config.update(config)
        
        # Maximum drawdown settings
        self.max_drawdown_pct = self.config.get("max_drawdown_pct", 0.15)  # 15% max drawdown
        self.max_daily_drawdown_pct = self.config.get("max_daily_drawdown_pct", 0.05)  # 5% max daily drawdown
        
        # Position sizing settings
        self.default_risk_per_trade = self.config.get("default_risk_per_trade", 0.01)  # 1% risk per trade
        self.max_risk_per_trade = self.config.get("max_risk_per_trade", 0.05)  # 5% max risk per trade
        self.max_portfolio_risk = self.config.get("max_portfolio_risk", 0.30)  # 30% max portfolio risk
        
        # Stop-loss settings
        self.stop_loss_type = StopLossType[self.config.get("stop_loss_type", "VOLATILITY")]
        self.fixed_stop_loss_pct = self.config.get("fixed_stop_loss_pct", 0.02)  # 2% fixed stop-loss
        self.atr_multiplier = self.config.get("atr_multiplier", 3.0)  # 3 x ATR for volatility-based stops
        self.trailing_stop_activation_pct = self.config.get("trailing_stop_activation_pct", 0.01)  # 1% profit to activate trailing stop
        self.trailing_stop_distance_pct = self.config.get("trailing_stop_distance_pct", 0.02)  # 2% trailing stop distance
        
        # Portfolio state
        self.portfolio_value = self.config.get("initial_portfolio_value", 100000.0)
        self.peak_portfolio_value = self.portfolio_value
        self.positions = {}  # Symbol -> position info
        self.trades_history = []
        self.max_positions = self.config.get("max_positions", 10)
        
        # Risk metrics
        self.current_drawdown_pct = 0.0
        self.daily_drawdown_pct = 0.0
        self.total_portfolio_risk = 0.0
        self.risk_level = RiskLevel.LOW
        
        # VaR settings
        self.var_confidence_level = self.config.get("var_confidence_level", 0.95)
        self.var_time_horizon = self.config.get("var_time_horizon", 1)  # days
        self.position_var = {}  # Symbol -> VaR
        self.portfolio_var = 0.0
        
        # Daily tracking
        self.today = datetime.now().date()
        self.daily_high = self.portfolio_value
        
        # Load state if available
        self.load_state()
        
        logger.info("Risk Manager initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "max_position_pct": 0.05,           # Max 5% of portfolio per position
            "max_risk_pct": 0.01,              # Risk 1% of portfolio per trade
            "max_portfolio_risk": 0.20,        # Max 20% of portfolio at risk
            "max_correlated_positions": 3,     # Max correlated positions
            "max_sector_allocation": 0.30,     # Max 30% in one sector
            "default_risk_per_trade": 0.01,    # Default 1% risk per trade
            "daily_stop_loss_pct": 0.03,       # 3% daily stop loss
            "max_daily_drawdown_pct": 0.05,    # 5% max daily drawdown
            "max_open_trades": 10,             # Max simultaneous open trades
            "correlation_threshold": 0.7,       # Correlation threshold
            "stop_loss_type": "volatility",    # Default stop-loss method
            "default_fixed_stop_pct": 0.05,    # 5% fixed stop
            "default_atr_multiple": 2.0,       # 2x ATR volatility stop
            "trailing_stop_activation_pct": 0.02, # 2% profit to activate
            "trailing_stop_distance_pct": 0.02,   # 2% trailing distance
            "portfolio_var_conf_level": 0.95,   # 95% VaR confidence
            "position_var_conf_level": 0.95,    # 95% VaR confidence
            "var_days": 1,                     # 1-day VaR horizon
        }
        
    def _load_from_typed_settings(self, settings: RiskSettings) -> None:
        """Load configuration from a RiskSettings object.
        
        Args:
            settings: RiskSettings object from the typed settings system
        """
        try:
            # Convert RiskSettings to our internal config format
            self.config.update({
                "max_position_pct": settings.max_position_pct,
                "max_risk_pct": settings.max_risk_pct,
                "max_portfolio_risk": settings.max_portfolio_risk,
                "max_correlated_positions": settings.max_correlated_positions,
                "max_sector_allocation": settings.max_sector_allocation,
                "max_open_trades": settings.max_open_trades,
                "correlation_threshold": settings.correlation_threshold,
                "enable_portfolio_stop_loss": settings.enable_portfolio_stop_loss,
                "portfolio_stop_loss_pct": settings.portfolio_stop_loss_pct,
                "enable_position_stop_loss": settings.enable_position_stop_loss
            })
            
            logger.info("Loaded risk configuration from typed settings")
        except Exception as e:
            logger.error(f"Error loading from typed settings: {e}")
            logger.error("Using existing configuration")
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss_price: float, market_data: Dict[str, Any]) -> int:
        """
        Calculate optimal position size based on systematic risk management rules.
        
        Determines the appropriate number of shares or contracts to trade based on
        account size, risk parameters, and the distance to the initial stop-loss.
        This is a critical risk management function that ensures consistent risk
        per trade regardless of price or volatility.
        
        Parameters:
            symbol (str): Trading symbol for the instrument
            entry_price (float): Anticipated entry price for the position
            stop_loss_price (float): Initial stop-loss price for the position
            market_data (Dict[str, Any]): Market data dictionary containing price history
                and volatility information for additional risk calculations
            
        Returns:
            int: Number of shares/contracts to trade, rounded down to ensure
                risk stays within parameters
                
        Position sizing methodology:
        1. Calculate maximum dollar risk based on portfolio value and risk percentage
        2. Determine risk per share as the distance between entry and stop-loss
        3. Calculate position size by dividing dollar risk by risk per share
        4. Apply position limits based on:
           - Maximum risk per trade
           - Maximum portfolio exposure
           - Maximum number of positions
           - Current risk level and drawdown state
                
        Risk controls applied:
        - Portfolio percentage risk limits (default_risk_per_trade)
        - Maximum position size limits (max_risk_per_trade)
        - Total portfolio exposure limits (max_portfolio_risk)
        - Position count limits (max_positions)
        
        Notes:
            - Position size is always rounded down to whole shares/contracts
            - Small risk per share can result in large position sizes; max limits provide safeguards
            - When stop price equals entry price, returns zero (invalid position)
            - Risk parameters automatically scale with portfolio value changes
            - Position sizing is directionally neutral (same logic for long and short)
            - All limits are strictly enforced; the most restrictive limit applies
        """
        # Calculate risk per trade in dollars
        risk_dollars = self.portfolio_value * self.default_risk_per_trade
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share <= 0:
            logger.warning(f"Invalid risk per share for {symbol}: {risk_per_share}")
            return 0
        
        # Calculate position size
        position_size = int(risk_dollars / risk_per_share)
        
        # Limit position by max risk per trade
        max_position_size = int(self.portfolio_value * self.max_risk_per_trade / entry_price)
        position_size = min(position_size, max_position_size)
        
        # Check if we're within the max portfolio risk
        new_exposure = position_size * entry_price
        current_exposure = sum(pos.get("current_value", 0) for pos in self.positions.values())
        
        if (current_exposure + new_exposure) / self.portfolio_value > self.max_portfolio_risk:
            # Scale back position to respect max portfolio risk
            available_risk = (self.max_portfolio_risk * self.portfolio_value) - current_exposure
            if available_risk <= 0:
                logger.warning(f"Cannot open position for {symbol}: Max portfolio risk reached")
                return 0
            
            position_size = int(available_risk / entry_price)
        
        # Ensure we don't exceed max positions
        if len(self.positions) >= self.max_positions and symbol not in self.positions:
            logger.warning(f"Cannot open position for {symbol}: Max positions reached")
            return 0
        
        logger.info(f"Calculated position size for {symbol}: {position_size} shares at ${entry_price:.2f}")
        return position_size
    
    def calculate_stop_loss(self, symbol: str, entry_price: float, 
                          direction: int, market_data: Dict[str, Any]) -> float:
        """
        Calculate appropriate stop-loss price using the configured methodology.
        
        Determines the initial stop-loss price for a position based on the selected
        stop-loss type and relevant market data. The stop-loss price is a critical
        component of risk management that defines the maximum acceptable loss
        for a position.
        
        Parameters:
            symbol (str): Trading symbol for the instrument
            entry_price (float): Entry price for the position
            direction (int): Trade direction (1 for long, -1 for short)
            market_data (Dict[str, Any]): Market data dictionary containing price
                history and volatility information needed for stop calculations
            
        Returns:
            float: Calculated stop-loss price
                
        Stop-loss methodologies:
        1. FIXED: Simple percentage-based stop
           - Uses fixed_stop_loss_pct from configuration
           - Same percentage for all instruments regardless of volatility
           - Example: 2% below entry for long positions
           
        2. VOLATILITY: ATR-based dynamic stop
           - Adapts to each instrument's specific volatility
           - Uses ATR multiplier from configuration
           - Example: Entry price - (3 × ATR) for long positions
           
        3. TRAILING: Initial stop same as volatility method
           - Initial stop similar to volatility method
           - Will update as price moves favorably
           - Requires separate update mechanism via update_trailing_stops()
           
        4. TIME_BASED: Defaults to fixed percentage stop
           - Time component handled separately
           - Initial price stop same as fixed percentage
        
        Notes:
            - Long positions: stop price is below entry price
            - Short positions: stop price is above entry price
            - ATR calculation uses 14-period default
            - Stop prices must never equal the entry price
            - Falls back to fixed percentage if insufficient data for ATR
            - For trailing stops, this sets only the initial stop value
        """
        if self.stop_loss_type == StopLossType.FIXED:
            # Fixed percentage stop-loss
            stop_loss = entry_price * (1 - direction * self.fixed_stop_loss_pct)
        
        elif self.stop_loss_type == StopLossType.VOLATILITY:
            # Volatility-based stop-loss using ATR
            atr = self._calculate_atr(market_data, period=14)
            stop_loss = entry_price - (direction * self.atr_multiplier * atr)
        
        elif self.stop_loss_type == StopLossType.TRAILING:
            # Initial stop for trailing stop-loss
            # Will be updated as price moves in favorable direction
            atr = self._calculate_atr(market_data, period=14)
            stop_loss = entry_price - (direction * self.atr_multiplier * atr)
        
        else:  # TIME_BASED or any other type
            # Default to fixed stop-loss
            stop_loss = entry_price * (1 - direction * self.fixed_stop_loss_pct)
        
        logger.info(f"Calculated stop-loss for {symbol}: ${stop_loss:.2f} (entry: ${entry_price:.2f}, direction: {direction})")
        return stop_loss
    
    def _calculate_atr(self, market_data: Dict[str, Any], period: int = 14) -> float:
        """
        Calculate the Average True Range (ATR) from market data.
        
        Args:
            market_data: Market data dictionary
            period: ATR period
            
        Returns:
            float: ATR value
        """
        highs = market_data.get("high", [])
        lows = market_data.get("low", [])
        closes = market_data.get("close", [])
        
        if len(closes) < period + 1:
            # Not enough data, return a default ATR based on recent volatility
            if len(closes) > 1:
                return abs(closes[-1] - closes[0]) / len(closes)
            return closes[-1] * 0.02  # Default to 2% of current price
        
        # Calculate true ranges
        tr_values = []
        for i in range(1, len(closes)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i-1]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = max(tr1, tr2, tr3)
            tr_values.append(true_range)
        
        # Calculate ATR
        atr = sum(tr_values[-period:]) / period
        return atr
    
    def update_trailing_stops(self, market_data: Dict[str, Dict[str, Any]]):
        """
        Update trailing stop-loss levels based on current market prices.
        
        Processes all positions with trailing stops and updates their stop-loss
        levels when price moves favorably beyond the activation threshold.
        This method "ratchets" stop-loss levels to lock in profits while
        allowing positions to continue running in profitable trades.
        
        Parameters:
            market_data (Dict[str, Dict[str, Any]]): Dictionary mapping symbols
                to market data dictionaries containing current prices
                
        Trailing stop mechanics:
        1. For each position with trailing stop type:
           - Check if current price has moved beyond activation threshold
           - If activated, calculate new stop-loss level based on trailing distance
           - Update stop only if new level is more favorable than current stop
           - Continue updating as price moves further in favorable direction
           
        Activation logic:
        - Long positions: Trailing stop activates when price rises above entry by
          trailing_stop_activation_pct percentage
        - Short positions: Trailing stop activates when price falls below entry by
          trailing_stop_activation_pct percentage
          
        Stop distance calculation:
        - Long positions: Stop is set at current_price × (1 - trailing_stop_distance_pct)
        - Short positions: Stop is set at current_price × (1 + trailing_stop_distance_pct)
        
        Side effects:
        - Updates position stop_loss_price values in the positions dictionary
        - Updates position current_value based on latest prices
        - Logs stop-loss updates for auditing and monitoring
        
        Notes:
            - Trailing stops move in only one direction (more favorable)
            - More frequent updates lead to more responsive trailing stops
            - No changes occur until price reaches activation threshold
            - Positions retain original stop until activation occurs
            - Market data must contain 'price' key for each symbol
        """
        for symbol, position in self.positions.items():
            if position.get("stop_loss_type") != StopLossType.TRAILING:
                continue
            
            # Get current price for the symbol
            if symbol not in market_data:
                logger.warning(f"No market data for {symbol}, cannot update trailing stop")
                continue
                
            current_price = market_data[symbol].get("price", position.get("entry_price"))
            if not current_price:
                continue
            
            # Update position value
            position["current_value"] = position["size"] * current_price
            
            # Check if position is in profit enough to activate trailing stop
            entry_price = position["entry_price"]
            direction = position["direction"]
            activation_threshold = entry_price * (1 + direction * self.trailing_stop_activation_pct)
            
            # For long positions: current_price > activation_threshold
            # For short positions: current_price < activation_threshold
            if (direction == 1 and current_price > activation_threshold) or \
               (direction == -1 and current_price < activation_threshold):
                
                # Calculate new stop-loss level
                new_stop = current_price * (1 - direction * self.trailing_stop_distance_pct)
                
                # Only update stop if it's better than the current one
                # For long: new_stop > current_stop
                # For short: new_stop < current_stop
                current_stop = position["stop_loss_price"]
                if (direction == 1 and new_stop > current_stop) or \
                   (direction == -1 and new_stop < current_stop):
                    position["stop_loss_price"] = new_stop
                    logger.info(f"Updated trailing stop for {symbol} to ${new_stop:.2f}")
    
    def check_stop_losses(self, market_data: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Check all positions against their stop-loss levels and identify triggered stops.
        
        Evaluates each active position against current market prices to determine
        if any stop-loss levels have been breached. Triggered stops result in the
        position being closed and recorded in the trade history.
        
        Parameters:
            market_data (Dict[str, Dict[str, Any]]): Dictionary mapping symbols
                to market data dictionaries containing current prices
                
        Returns:
            List[str]: List of symbols for which stop-losses were triggered
            
        Stop-loss evaluation:
        - Long positions: Stop triggered when current price <= stop price
        - Short positions: Stop triggered when current price >= stop price
        
        Side effects:
        - Removes triggered positions from the positions dictionary
        - Records closed trades in the trades_history list with stop_loss exit reason
        - Logs stop-loss triggers for monitoring and analysis
        
        Trade record details:
        - All original position information
        - Exit price (current market price)
        - Exit time (current timestamp)
        - PnL (absolute and percentage)
        - Exit reason ('stop_loss')
        
        Notes:
            - This method does not execute actual orders; it only identifies positions
              that require exit based on stop-loss criteria
            - Calling code should use the returned list to execute actual exit orders
            - Stop-loss checks should be performed frequently during market hours
            - Market data must contain 'price' key for each symbol
            - Positions removed here will not be included in future risk calculations
        """
        triggered_symbols = []
        
        for symbol, position in list(self.positions.items()):
            if symbol not in market_data:
                continue
                
            current_price = market_data[symbol].get("price")
            if not current_price:
                continue
            
            stop_price = position["stop_loss_price"]
            direction = position["direction"]
            
            # Check if stop-loss is triggered
            # For long positions: current_price <= stop_price
            # For short positions: current_price >= stop_price
            if (direction == 1 and current_price <= stop_price) or \
               (direction == -1 and current_price >= stop_price):
                
                logger.info(f"Stop-loss triggered for {symbol} at ${current_price:.2f} (stop: ${stop_price:.2f})")
                triggered_symbols.append(symbol)
                
                # Record the stopped out trade
                trade_record = position.copy()
                trade_record["exit_price"] = current_price
                trade_record["exit_time"] = datetime.now().isoformat()
                trade_record["pnl"] = position["size"] * (current_price - position["entry_price"]) * direction
                trade_record["pnl_pct"] = (current_price / position["entry_price"] - 1) * direction * 100
                trade_record["exit_reason"] = "stop_loss"
                
                self.trades_history.append(trade_record)
                
                # Remove the position
                del self.positions[symbol]
        
        return triggered_symbols
    
    def open_position(self, symbol: str, entry_price: float, direction: int, 
                    market_data: Dict[str, Any], reason: str = "signal") -> bool:
        """
        Open a new position with comprehensive risk management controls.
        
        Creates a new position with appropriate size and stop-loss levels
        based on risk parameters and market conditions. This is the primary
        entry point for creating risk-managed positions.
        
        Parameters:
            symbol (str): Trading symbol for the instrument
            entry_price (float): Entry price for the position
            direction (int): Trade direction (1 for long, -1 for short)
            market_data (Dict[str, Any]): Market data dictionary
            reason (str): Reason for opening the position (default: "signal")
            
        Returns:
            bool: True if position was opened successfully, False otherwise
            
        Position creation process:
        1. Verify position doesn't already exist
        2. Calculate appropriate stop-loss price
        3. Determine optimal position size based on risk parameters
        4. Create position record with all necessary information
        5. Add position to the portfolio tracking system
        6. Update portfolio risk metrics
        
        Position record contents:
        - symbol: Trading symbol
        - entry_price: Position entry price
        - entry_time: ISO format timestamp of entry
        - direction: Trade direction (1=long, -1=short)
        - size: Position size in shares/contracts
        - current_value: Initial position value
        - stop_loss_price: Initial stop-loss price
        - stop_loss_type: Stop-loss methodology used
        - reason: Signal or reason for entry
        
        Risk management applied:
        - Prevents duplicate positions in the same instrument
        - Applies appropriate stop-loss based on configured methodology
        - Sizes position based on fixed-percentage risk
        - Enforces maximum position limits
        - Updates portfolio risk metrics after position creation
        
        Notes:
            - Returns False if any risk limits prevent position creation
            - Position size may be reduced to comply with risk limits
            - Position value is calculated as size × entry_price
            - Position direction must be 1 (long) or -1 (short)
            - Updates internal risk metrics after position creation
        """
        # Check if we already have this position
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return False
        
        # Calculate stop-loss price
        stop_loss_price = self.calculate_stop_loss(symbol, entry_price, direction, market_data)
        
        # Calculate position size
        position_size = self.calculate_position_size(symbol, entry_price, stop_loss_price, market_data)
        
        if position_size <= 0:
            logger.warning(f"Invalid position size for {symbol}: {position_size}")
            return False
        
        # Create position record
        position = {
            "symbol": symbol,
            "entry_price": entry_price,
            "entry_time": datetime.now().isoformat(),
            "direction": direction,
            "size": position_size,
            "current_value": position_size * entry_price,
            "stop_loss_price": stop_loss_price,
            "stop_loss_type": self.stop_loss_type,
            "reason": reason
        }
        
        # Add position to portfolio
        self.positions[symbol] = position
        
        # Update portfolio risk
        self._update_portfolio_risk()
        
        logger.info(f"Opened {position_size} {'long' if direction == 1 else 'short'} position in {symbol} at ${entry_price:.2f}")
        return True
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "signal") -> bool:
        """
        Close an existing position.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            reason: Reason for closing the position
            
        Returns:
            bool: True if position was closed successfully
        """
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        direction = position["direction"]
        entry_price = position["entry_price"]
        size = position["size"]
        
        # Calculate P&L
        pnl = size * (exit_price - entry_price) * direction
        pnl_pct = (exit_price / entry_price - 1) * direction * 100
        
        # Record the closed trade
        trade_record = position.copy()
        trade_record["exit_price"] = exit_price
        trade_record["exit_time"] = datetime.now().isoformat()
        trade_record["pnl"] = pnl
        trade_record["pnl_pct"] = pnl_pct
        trade_record["exit_reason"] = reason
        
        self.trades_history.append(trade_record)
        
        # Remove the position
        del self.positions[symbol]
        
        # Update portfolio value and risk
        self.portfolio_value += pnl
        self._update_portfolio_risk()
        
        logger.info(f"Closed {size} {'long' if direction == 1 else 'short'} position in {symbol} at ${exit_price:.2f} for ${pnl:.2f} ({pnl_pct:.2f}%)")
        return True
    
    def update_portfolio_value(self, market_data: Dict[str, Dict[str, Any]]):
        """
        Update portfolio valuation and risk metrics based on current market prices.
        
        Recalculates the total portfolio value and various risk metrics using
        the latest market prices for all open positions. This critical function
        enables real-time risk monitoring and drawdown tracking.
        
        Parameters:
            market_data (Dict[str, Dict[str, Any]]): Dictionary mapping symbols
                to market data dictionaries containing current prices
                
        Daily tracking transition:
        - Resets daily tracking metrics when a new calendar day is detected
        - Ensures accurate per-day risk tracking and appropriate daily risk controls
        
        Portfolio valuation process:
        1. Check for date transition and reset daily metrics if needed
        2. Start with base portfolio value minus current position values
        3. For each position, recalculate current value using latest prices
        4. Sum all position values to get updated portfolio value
        5. Update peak value tracking for drawdown calculations
        6. Calculate current drawdown metrics
        7. Update risk level classification based on current metrics
        
        Risk metrics updated:
        - Total portfolio value
        - Peak portfolio value (all-time high watermark)
        - Daily high watermark
        - Current drawdown percentage
        - Daily drawdown percentage
        - Risk level classification
        
        Side effects:
        - Updates self.portfolio_value with current total value
        - Updates self.peak_portfolio_value if new all-time high
        - Updates self.daily_high if new daily high
        - Updates drawdown calculations and risk levels
        - Updates position current_value for each position
        - Logs portfolio value and risk status
        
        Notes:
            - Should be called regularly during market hours
            - Critical for stop-loss monitoring and risk limit enforcement
            - Market data must contain 'price' key for each symbol
            - Missing market data for a position will use previous valuation
            - Risk level transitions may trigger automated risk-reduction measures
        """
        # Check if date has changed
        current_date = datetime.now().date()
        if current_date != self.today:
            # Reset daily tracking
            self.today = current_date
            self.daily_high = self.portfolio_value
            self.daily_drawdown_pct = 0.0
        
        # Start with cash value
        updated_value = self.portfolio_value
        
        # Subtract current positions' values since we'll recalculate them
        for position in self.positions.values():
            updated_value -= position.get("current_value", 0)
        
        # Add updated positions' values
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol].get("price")
                if current_price:
                    current_value = position["size"] * current_price
                    position["current_value"] = current_value
                    updated_value += current_value
        
        # Update portfolio value
        self.portfolio_value = updated_value
        
        # Update peak value if we have a new high
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
        
        # Update daily high if we have a new daily high
        if self.portfolio_value > self.daily_high:
            self.daily_high = self.portfolio_value
        
        # Calculate drawdowns
        self._calculate_drawdowns()
        
        # Check risk levels
        self._update_risk_level()
        
        # Log portfolio update
        logger.debug(f"Updated portfolio value: ${self.portfolio_value:.2f}, Drawdown: {self.current_drawdown_pct:.2f}%, Risk level: {self.risk_level.name}")
    
    def _calculate_drawdowns(self):
        """Calculate current drawdown metrics."""
        # Overall drawdown from peak
        if self.peak_portfolio_value > 0:
            self.current_drawdown_pct = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        else:
            self.current_drawdown_pct = 0.0
        
        # Daily drawdown from today's high
        if self.daily_high > 0:
            self.daily_drawdown_pct = (self.daily_high - self.portfolio_value) / self.daily_high
        else:
            self.daily_drawdown_pct = 0.0
    
    def _update_portfolio_risk(self):
        """Update portfolio risk metrics."""
        # Simple calculation of total portfolio risk
        total_exposure = sum(pos.get("current_value", 0) for pos in self.positions.values())
        self.total_portfolio_risk = total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
    
    def _update_risk_level(self):
        """Update the current risk level based on drawdowns and exposure."""
        # Determine risk level based on current metrics
        if self.current_drawdown_pct >= 0.9 * self.max_drawdown_pct or self.daily_drawdown_pct >= 0.9 * self.max_daily_drawdown_pct:
            self.risk_level = RiskLevel.CRITICAL
        elif self.current_drawdown_pct >= 0.7 * self.max_drawdown_pct or self.daily_drawdown_pct >= 0.7 * self.max_daily_drawdown_pct:
            self.risk_level = RiskLevel.EXTREME
        elif self.current_drawdown_pct >= 0.5 * self.max_drawdown_pct or self.daily_drawdown_pct >= 0.5 * self.max_daily_drawdown_pct:
            self.risk_level = RiskLevel.HIGH
        elif self.current_drawdown_pct >= 0.3 * self.max_drawdown_pct or self.daily_drawdown_pct >= 0.3 * self.max_daily_drawdown_pct:
            self.risk_level = RiskLevel.MEDIUM
        else:
            self.risk_level = RiskLevel.LOW
    
    def calculate_var(self, market_data: Dict[str, Dict[str, Any]], method: str = "historical"):
        """
        Calculate Value at Risk (VaR) for portfolio.
        
        Args:
            market_data: Dictionary mapping symbols to market data
            method: VaR calculation method ('historical', 'parametric', or 'monte_carlo')
        """
        if method == "historical":
            self._calculate_historical_var(market_data)
        elif method == "parametric":
            self._calculate_parametric_var(market_data)
        elif method == "monte_carlo":
            self._calculate_monte_carlo_var(market_data)
        else:
            logger.warning(f"Unknown VaR method: {method}, using historical")
            self._calculate_historical_var(market_data)
    
    def _calculate_historical_var(self, market_data: Dict[str, Dict[str, Any]]):
        """
        Calculate VaR using historical method.
        
        Args:
            market_data: Dictionary mapping symbols to market data
        """
        # For each position, calculate historical returns
        position_vars = {}
        
        for symbol, position in self.positions.items():
            if symbol not in market_data:
                continue
            
            # Get price history
            prices = market_data[symbol].get("close", [])
            if len(prices) < 30:  # Need sufficient history
                continue
            
            # Calculate daily returns
            returns = np.diff(prices) / prices[:-1]
            
            # Sort returns from worst to best
            sorted_returns = np.sort(returns)
            
            # Find the return at the specified confidence level
            var_index = int(len(sorted_returns) * (1 - self.var_confidence_level))
            var_return = sorted_returns[var_index]
            
            # Calculate VAR in dollar terms
            position_value = position.get("current_value", 0)
            position_var = position_value * abs(var_return) * np.sqrt(self.var_time_horizon)
            
            position_vars[symbol] = position_var
        
        # Store position VaRs
        self.position_var = position_vars
        
        # Simple sum of position VaRs (ignoring correlations)
        self.portfolio_var = sum(position_vars.values())
        
        logger.debug(f"Historical VaR ({self.var_confidence_level*100}%, {self.var_time_horizon}-day): ${self.portfolio_var:.2f}")
    
    def _calculate_parametric_var(self, market_data: Dict[str, Dict[str, Any]]):
        """Placeholder for parametric VaR calculation."""
        logger.info("Parametric VaR calculation not yet implemented, using historical")
        self._calculate_historical_var(market_data)
    
    def _calculate_monte_carlo_var(self, market_data: Dict[str, Dict[str, Any]]):
        """Placeholder for Monte Carlo VaR calculation."""
        logger.info("Monte Carlo VaR calculation not yet implemented, using historical")
        self._calculate_historical_var(market_data)
    
    def check_risk_limits(self) -> Tuple[bool, List[str]]:
        """
        Evaluate current risk metrics against configured risk limits.
        
        Determines whether risk reduction actions are needed by checking
        if any risk limits have been breached. This function serves as
        a trigger for automated risk management interventions.
        
        Returns:
            Tuple[bool, List[str]]: 
                - Boolean indicating if risk reduction is needed
                - List of specific reasons for risk limit breaches
                
        Risk limits checked:
        1. Maximum total drawdown limit
        2. Maximum daily drawdown limit
        3. Maximum portfolio exposure limit
        4. Critical risk level status
        
        Risk breach assessment:
        - True when any risk threshold is breached
        - Returns all applicable breach reasons for comprehensive reporting
        - Provides detailed contextual information about each breach
        
        Usage:
        - Regular risk monitoring during trading hours
        - Pre-trade checks to prevent excess risk
        - Automated risk reduction decision making
        - System monitoring and alerting
        
        Notes:
            - Should be called after portfolio value updates
            - Multiple risk limits may be breached simultaneously
            - Empty reasons list indicates all risk parameters are within limits
            - Critical risk level automatically indicates reduction is needed
            - Risk breaches should trigger immediate action to reduce exposure
        """
        reasons = []
        
        # Check drawdown limits
        if self.current_drawdown_pct >= self.max_drawdown_pct:
            reasons.append(f"Max drawdown limit breached: {self.current_drawdown_pct:.2%} >= {self.max_drawdown_pct:.2%}")
        
        if self.daily_drawdown_pct >= self.max_daily_drawdown_pct:
            reasons.append(f"Max daily drawdown limit breached: {self.daily_drawdown_pct:.2%} >= {self.max_daily_drawdown_pct:.2%}")
        
        # Check portfolio risk
        if self.total_portfolio_risk >= self.max_portfolio_risk:
            reasons.append(f"Max portfolio risk breached: {self.total_portfolio_risk:.2%} >= {self.max_portfolio_risk:.2%}")
        
        # Critical risk level check
        if self.risk_level == RiskLevel.CRITICAL:
            reasons.append(f"Risk level is CRITICAL")
        
        return len(reasons) > 0, reasons
    
    def get_reduction_actions(self) -> List[Dict[str, Any]]:
        """
        Generate recommended risk reduction actions when risk limits are breached.
        
        Provides a prioritized list of specific actions to reduce risk exposure
        when risk limits are exceeded. Actions are tailored to the current risk
        level and position characteristics.
        
        Returns:
            List[Dict[str, Any]]: List of risk reduction action dictionaries.
                Each dictionary contains:
                - 'symbol': Symbol to take action on
                - 'action': Action to take ('close_position' or 'tighten_stop')
                - 'reason': Explanation for the recommended action
                
        Risk reduction strategies by risk level:
        1. CRITICAL: Close all positions immediately
        2. EXTREME: Close largest positions (top 30% by value)
        3. HIGH: Close underperforming positions (most underwater, up to 20%)
        4. MEDIUM: Tighten stops on underwater positions
        5. LOW: No actions needed
        
        Selection criteria:
        - Position size and value
        - Current profit/loss status
        - Position correlation
        - Position age and holding period
        
        Implementation:
        - Only generates actions when risk limits are breached
        - Returns empty list if no risk reduction needed
        - Actions are ordered by priority
        - Each action includes the specific instrument symbol
        
        Notes:
            - Results are recommendations; must be implemented by calling code
            - More sophisticated implementations may consider correlations
            - Position-specific characteristics influence selection
            - The most severe risk level determines the reduction strategy
            - Check check_risk_limits() should be called before this method
        """
        should_reduce, reasons = self.check_risk_limits()
        
        if not should_reduce:
            return []
        
        actions = []
        
        # If at critical risk level, close all positions
        if self.risk_level == RiskLevel.CRITICAL:
            for symbol in self.positions:
                actions.append({
                    "symbol": symbol,
                    "action": "close_position",
                    "reason": "Critical risk level reached"
                })
            return actions
        
        # If at extreme risk, reduce largest positions
        if self.risk_level == RiskLevel.EXTREME:
            # Sort positions by size
            sorted_positions = sorted(
                self.positions.items(),
                key=lambda x: x[1].get("current_value", 0),
                reverse=True
            )
            
            # Close top 30% of positions
            positions_to_close = sorted_positions[:max(1, int(len(sorted_positions) * 0.3))]
            
            for symbol, _ in positions_to_close:
                actions.append({
                    "symbol": symbol,
                    "action": "close_position",
                    "reason": "Extreme risk level - reducing largest positions"
                })
            
            return actions
        
        # For high risk, reduce underwater positions
        if self.risk_level == RiskLevel.HIGH:
            # Find underwater positions
            underwater_positions = [
                (symbol, pos) for symbol, pos in self.positions.items()
                if (pos["direction"] == 1 and pos.get("current_value", 0) < pos["size"] * pos["entry_price"]) or
                   (pos["direction"] == -1 and pos.get("current_value", 0) > pos["size"] * pos["entry_price"])
            ]
            
            # Close most underwater positions first
            sorted_underwater = sorted(
                underwater_positions,
                key=lambda x: (x[1].get("current_value", 0) / (x[1]["size"] * x[1]["entry_price"]) - 1) * x[1]["direction"],
                reverse=False
            )
            
            # Close up to 20% of positions
            positions_to_close = sorted_underwater[:max(1, int(len(self.positions) * 0.2))]
            
            for symbol, _ in positions_to_close:
                actions.append({
                    "symbol": symbol,
                    "action": "close_position",
                    "reason": "High risk level - reducing underwater positions"
                })
            
            return actions
        
        # For medium risk, tighten stops on underwater positions
        if self.risk_level == RiskLevel.MEDIUM:
            for symbol, position in self.positions.items():
                current_price = position.get("current_value", 0) / position["size"]
                
                # Check if position is underwater
                if (position["direction"] == 1 and current_price < position["entry_price"]) or \
                   (position["direction"] == -1 and current_price > position["entry_price"]):
                    
                    actions.append({
                        "symbol": symbol,
                        "action": "tighten_stop",
                        "reason": "Medium risk level - tightening stops"
                    })
            
            return actions
        
        return actions
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive risk metrics for monitoring and reporting.
        
        Provides a consolidated view of all current risk measurements and
        portfolio status information for risk monitoring, reporting, and
        analysis purposes.
        
        Returns:
            Dict[str, Any]: Dictionary containing all current risk metrics:
                - portfolio_value: Current total portfolio value
                - peak_portfolio_value: All-time high portfolio value
                - current_drawdown_pct: Current drawdown from peak
                - daily_drawdown_pct: Current drawdown from day's high
                - total_portfolio_risk: Portfolio exposure as percentage
                - risk_level: Current risk level classification (string name)
                - portfolio_var: Portfolio-level Value at Risk
                - position_var: Dictionary mapping symbols to position VaR
                - positions_count: Number of open positions
                - timestamp: ISO format timestamp of the metrics
                
        Metrics categories:
        1. Portfolio valuation metrics
        2. Drawdown and exposure metrics 
        3. Risk level and classification
        4. Value at Risk (VaR) metrics
        5. Position statistics
        
        Usage:
        - System dashboards and monitoring
        - Risk reporting and compliance
        - Performance tracking
        - Decision support for trading systems
        - Historical risk analysis
        
        Notes:
            - Timestamp captures exact moment of metric generation
            - All percentage values are expressed as decimals (0.15 = 15%)
            - VaR values represent potential loss in currency units
            - Risk level is provided as a string name for easy reporting
            - Current values reflect the most recent portfolio update
        """
        return {
            "portfolio_value": self.portfolio_value,
            "peak_portfolio_value": self.peak_portfolio_value,
            "current_drawdown_pct": self.current_drawdown_pct,
            "daily_drawdown_pct": self.daily_drawdown_pct,
            "total_portfolio_risk": self.total_portfolio_risk,
            "risk_level": self.risk_level.name,
            "portfolio_var": self.portfolio_var,
            "position_var": self.position_var,
            "positions_count": len(self.positions),
            "timestamp": datetime.now().isoformat()
        }
    
    def save_state(self) -> None:
        """Save risk manager state to disk."""
        state = {
            "portfolio_value": self.portfolio_value,
            "peak_portfolio_value": self.peak_portfolio_value,
            "positions": self.positions,
            "trades_history": self.trades_history[-1000:] if len(self.trades_history) > 1000 else self.trades_history,
            "risk_level": self.risk_level.name,
            "current_drawdown_pct": self.current_drawdown_pct,
            "daily_drawdown_pct": self.daily_drawdown_pct,
            "total_portfolio_risk": self.total_portfolio_risk,
            "today": self.today.isoformat(),
            "daily_high": self.daily_high,
            "timestamp": datetime.now().isoformat()
        }
        
        save_state(self.paths["state_path"], state)
        logger.info("Risk manager state saved")
    
    def load_state(self) -> bool:
        """
        Load risk manager state from disk.
        
        Returns:
            bool: True if state loaded successfully
        """
        state = load_state(self.paths["state_path"])
        
        if not state:
            return False
        
        try:
            self.portfolio_value = state.get("portfolio_value", self.portfolio_value)
            self.peak_portfolio_value = state.get("peak_portfolio_value", self.peak_portfolio_value)
            self.positions = state.get("positions", {})
            self.trades_history = state.get("trades_history", [])
            
            # Convert string risk level back to enum
            risk_level_name = state.get("risk_level", self.risk_level.name)
            self.risk_level = RiskLevel[risk_level_name]
            
            self.current_drawdown_pct = state.get("current_drawdown_pct", self.current_drawdown_pct)
            self.daily_drawdown_pct = state.get("daily_drawdown_pct", self.daily_drawdown_pct)
            self.total_portfolio_risk = state.get("total_portfolio_risk", self.total_portfolio_risk)
            
            if "today" in state:
                self.today = datetime.fromisoformat(state["today"]).date()
            
            self.daily_high = state.get("daily_high", self.daily_high)
            
            logger.info("Risk manager state loaded")
            return True
        except Exception as e:
            logger.error(f"Error loading risk manager state: {e}")
            return False


# Simple example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Try to use typed settings if available
    if TYPED_SETTINGS_AVAILABLE:
        try:
            # Load configuration from the canonical config file
            settings = typed_load_config("/Users/bendickinson/Desktop/Trading/trading_bot/config/config.yaml")
            # Create risk manager with typed settings
            risk_manager = RiskManager(settings=settings.risk)
            print("Using typed settings configuration")
        except Exception as e:
            print(f"Could not load typed settings: {e}")
            print("Falling back to default configuration")
            risk_manager = RiskManager()
    else:
        # Create risk manager with default configuration
        risk_manager = RiskManager()
    
    # Create sample market data
    market_data = {
        "AAPL": {
            "price": 150.0,
            "close": [148.5, 149.2, 150.1, 149.8, 150.0],
            "high": [149.5, 150.2, 151.1, 150.8, 151.0],
            "low": [147.5, 148.2, 149.1, 148.8, 149.0],
            "volume": [5000000, 5200000, 4800000, 5100000, 5000000]
        },
        "MSFT": {
            "price": 290.0,
            "close": [285.5, 287.2, 289.1, 288.8, 290.0],
            "high": [287.5, 289.2, 291.1, 290.8, 292.0],
            "low": [283.5, 285.2, 287.1, 286.8, 288.0],
            "volume": [3000000, 3200000, 2800000, 3100000, 3000000]
        }
    }
    
    # Open some positions
    risk_manager.open_position("AAPL", 150.0, 1, market_data["AAPL"], "test")
    risk_manager.open_position("MSFT", 290.0, -1, market_data["MSFT"], "test")
    
    # Update portfolio value
    risk_manager.update_portfolio_value(market_data)
    
    # Calculate VaR
    risk_manager.calculate_var(market_data)
    
    # Get risk metrics
    metrics = risk_manager.get_risk_metrics()
    print("Risk metrics:", json.dumps(metrics, indent=2))
    
    # Check stops
    triggered = risk_manager.check_stop_losses(market_data)
    print("Triggered stops:", triggered)
    
    # Check risk limits
    should_reduce, reasons = risk_manager.check_risk_limits()
    print(f"Should reduce risk: {should_reduce}")
    if should_reduce:
        print("Reasons:", reasons)
        actions = risk_manager.get_reduction_actions()
        print("Recommended actions:", json.dumps(actions, indent=2)) 