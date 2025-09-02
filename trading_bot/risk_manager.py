"""
Multi-Asset Risk Manager Module

This module provides advanced risk management capabilities for trading across
multiple asset classes, integrating with the MultiAssetAdapter.
"""

import logging
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum

# Import the anomaly risk handler
from risk.anomaly_risk_handler import AnomalyRiskHandler

logger = logging.getLogger(__name__)

class RiskLevel(str, Enum):
    """Risk levels for trading decisions"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"

class RiskManager:
    """
    Risk management system for trading across multiple asset classes.
    Integrates with the MultiAssetAdapter to provide unified risk controls.
    """
    
    def __init__(self, 
                 multi_asset_adapter, 
                 config_path: Optional[str] = None, 
                 journal_dir: str = "journal",
                 anomaly_config_path: Optional[str] = None):
        """
        Initialize the risk manager.
        
        Args:
            multi_asset_adapter: MultiAssetAdapter instance
            config_path: Path to risk configuration file
            journal_dir: Directory for risk journal files
            anomaly_config_path: Path to anomaly risk configuration file
        """
        self.adapter = multi_asset_adapter
        self.config = self._load_config(config_path)
        self.journal_dir = journal_dir
        
        # Ensure journal directory exists
        os.makedirs(journal_dir, exist_ok=True)
        
        # Risk limits per asset class
        self.asset_class_limits = self.config.get("asset_class_limits", {})
        self.risk_profile = self.config.get("risk_profile", {})
        self.global_risk_limits = self.config.get("global_limits", {})
        
        # Position tracking
        self.open_trades = {}
        self.risk_history = []
        self.daily_risk_usage = {}
        
        # Portfolio metrics
        self.portfolio_metrics = {
            "total_exposure": 0,
            "exposure_by_asset": {},
            "exposure_by_class": {},
            "correlation_matrix": pd.DataFrame(),
            "max_drawdown_pct": 0,
            "current_drawdown_pct": 0,
            "heat_mapping": {}
        }
        
        # Initialize anomaly risk handler if anomaly detection is enabled
        self.anomaly_risk_handler = None
        self.anomaly_detection_enabled = self.config.get("enable_anomaly_detection", True)
        
        if self.anomaly_detection_enabled:
            self.anomaly_risk_handler = AnomalyRiskHandler(anomaly_config_path)
            logger.info("Anomaly risk handler initialized")
        
        # Track active risk modifiers due to anomalies
        self.active_risk_modifiers = {
            "position_size_modifier": 1.0,
            "stop_loss_modifier": 1.0
        }
        self.in_anomaly_cooldown = False
        self.anomaly_cooldown_end = None
        
        logger.info("Risk Manager initialized")
        
        # Load open trades from journal if available
        self._load_open_trades()
        
        # Immediately calculate portfolio metrics
        self.update_portfolio_metrics()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load risk configuration from file or use defaults."""
        if not config_path or not os.path.exists(config_path):
            logger.warning(f"Risk config not found at {config_path}, using defaults")
            return self._get_default_config()
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded risk configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load risk config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default risk configuration."""
        return {
            "risk_profile": {
                "profile_name": "balanced",
                "max_risk_per_trade": 1.0,  # % of account per trade
                "max_daily_risk": 3.0,      # % of account per day
                "max_portfolio_risk": 15.0, # % of account across all positions
                "position_size_multiplier": 1.0,
                "current_regime": "neutral" # market regime: bullish, neutral, bearish
            },
            "global_limits": {
                "max_drawdown": 15.0,       # % maximum drawdown before reducing risk
                "severe_drawdown": 25.0,    # % drawdown before stopping trading
                "max_leverage": 2.0,        # maximum account leverage
                "max_correlated_positions": 3, # maximum number of highly correlated positions
                "correlation_threshold": 0.7,  # correlation threshold for risk control
                "vix_risk_adjustment": True,   # adjust risk based on VIX level
                "max_open_positions": 20,      # maximum number of open positions
                "min_reward_risk_ratio": 1.5   # minimum reward:risk ratio for new trades
            },
            "asset_class_limits": {
                "equity": {
                    "max_allocation": 60.0,    # % of portfolio
                    "max_single_position": 5.0, # % of portfolio in single position
                    "max_sector_allocation": 30.0, # % of portfolio in single sector
                    "sector_correlations": True    # account for sector correlations
                },
                "futures": {
                    "max_allocation": 50.0,
                    "max_single_position": 10.0,
                    "max_sector_allocation": 25.0,
                    "intraday_margin_requirement": 1.0, # intraday margin multiplier
                    "overnight_margin_requirement": 1.5  # overnight margin multiplier
                },
                "forex": {
                    "max_allocation": 40.0,
                    "max_single_position": 5.0,
                    "max_correlated_pairs": 2,
                    "max_leverage": 10.0
                },
                "crypto": {
                    "max_allocation": 30.0,
                    "max_single_position": 5.0,
                    "max_leverage": 3.0,
                    "high_volatility_adjustment": 0.7  # reduce position size in high volatility
                },
                "options": {
                    "max_allocation": 15.0,
                    "max_single_position": 3.0,
                    "max_theta_exposure": 5.0,  # % of account in daily theta
                    "max_gamma_exposure": 1.0,  # normalized gamma risk
                    "max_vega_exposure": 2.0    # % of account exposed to 1% vol change
                }
            }
        }
    
    def _load_open_trades(self) -> None:
        """Load open trades from the journal."""
        journal_file = os.path.join(self.journal_dir, "open_trades.json")
        if os.path.exists(journal_file):
            try:
                with open(journal_file, 'r') as f:
                    self.open_trades = json.load(f)
                logger.info(f"Loaded {len(self.open_trades)} open trades from journal")
            except Exception as e:
                logger.error(f"Failed to load open trades: {e}")
    
    def _save_open_trades(self) -> None:
        """Save open trades to the journal."""
        journal_file = os.path.join(self.journal_dir, "open_trades.json")
        try:
            with open(journal_file, 'w') as f:
                json.dump(self.open_trades, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save open trades: {e}")
    
    def update_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Update portfolio risk metrics based on current positions.
        
        Returns:
            Dictionary of updated portfolio metrics
        """
        try:
            # Get current positions from adapter
            positions = self.adapter.get_positions()
            account_info = self.adapter.get_account_info()
            
            # Extract account value
            account_value = account_info.get("equity", account_info.get("balance", 0))
            if account_value <= 0:
                logger.error("Invalid account value")
                return self.portfolio_metrics
                
            # Reset metrics
            exposure_by_asset = {}
            exposure_by_class = {}
            total_exposure = 0
            
            # Process each position
            for position in positions:
                symbol = position.get("symbol", "")
                asset_class = position.get("asset_class", "unknown")
                
                # Calculate position value
                position_value = position.get("market_value", 0)
                if position_value == 0:
                    # Try alternate fields or calculate
                    quantity = abs(position.get("quantity", 0))
                    price = position.get("current_price", 0)
                    position_value = quantity * price
                
                # Add to exposure tracking
                exposure_by_asset[symbol] = position_value
                exposure_by_class[asset_class] = exposure_by_class.get(asset_class, 0) + position_value
                total_exposure += position_value
                
                # Update open trade record if we have it
                if symbol in self.open_trades:
                    self.open_trades[symbol].update({
                        "current_price": position.get("current_price", 0),
                        "current_value": position_value,
                        "unrealized_pl": position.get("unrealized_pl", 0),
                        "unrealized_pl_pct": position.get("unrealized_pl_pct", 0)
                    })
            
            # Update portfolio metrics
            self.portfolio_metrics.update({
                "total_exposure": total_exposure,
                "total_exposure_pct": (total_exposure / account_value) * 100 if account_value > 0 else 0,
                "exposure_by_asset": exposure_by_asset,
                "exposure_by_class": exposure_by_class,
                "leverage": total_exposure / account_value if account_value > 0 else 0,
                "open_positions_count": len(positions),
                "account_value": account_value
            })
            
            # Check drawdown if account history available
            if hasattr(self, "account_history") and self.account_history:
                peak_equity = max(h.get("equity", 0) for h in self.account_history)
                if peak_equity > 0 and account_value < peak_equity:
                    current_drawdown = (peak_equity - account_value) / peak_equity * 100
                    self.portfolio_metrics["current_drawdown_pct"] = current_drawdown
                    self.portfolio_metrics["max_drawdown_pct"] = max(
                        self.portfolio_metrics.get("max_drawdown_pct", 0),
                        current_drawdown
                    )
            
            # Update correlation matrix if we have historical data
            if len(exposure_by_asset) > 1:
                self._update_correlation_matrix(list(exposure_by_asset.keys()))
                
            # Risk heat map calculation
            self._calculate_risk_heat_map()
            
            logger.debug(f"Updated portfolio metrics: {len(positions)} positions, {total_exposure:.2f} exposure")
            return self.portfolio_metrics
            
        except Exception as e:
            logger.error(f"Failed to update portfolio metrics: {e}")
            return self.portfolio_metrics
    
    def _update_correlation_matrix(self, symbols: List[str]) -> None:
        """
        Update correlation matrix for position risk analysis.
        
        Args:
            symbols: List of symbols to analyze
        """
        try:
            if not symbols or len(symbols) < 2:
                return
                
            # Get price data for correlation calculation
            symbol_data = {}
            for symbol in symbols:
                data = self.adapter.get_data(
                    symbol=symbol,
                    timeframe="1d",
                    limit=60  # ~3 months of trading days
                )
                if not data.empty and 'close' in data.columns:
                    symbol_data[symbol] = data['close']
            
            # Create DataFrame with closing prices
            if len(symbol_data) < 2:
                return
                
            price_df = pd.DataFrame(symbol_data)
            
            # Calculate correlation matrix
            self.portfolio_metrics["correlation_matrix"] = price_df.corr()
            
            # Identify highly correlated pairs
            corr_matrix = self.portfolio_metrics["correlation_matrix"]
            threshold = self.global_risk_limits.get("correlation_threshold", 0.7)
            
            highly_correlated = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    symbol1 = corr_matrix.columns[i]
                    symbol2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    if abs(corr_value) >= threshold:
                        highly_correlated.append({
                            "symbol1": symbol1,
                            "symbol2": symbol2,
                            "correlation": corr_value
                        })
            
            self.portfolio_metrics["highly_correlated_pairs"] = highly_correlated
            
        except Exception as e:
            logger.error(f"Failed to update correlation matrix: {e}")
    
    def _calculate_risk_heat_map(self) -> None:
        """Calculate risk heat map for portfolio analysis."""
        heat_map = {}
        
        # Get exposure and limits
        exposure_by_class = self.portfolio_metrics.get("exposure_by_class", {})
        account_value = self.portfolio_metrics.get("account_value", 0)
        
        # Calculate usage percentage for each asset class
        for asset_class, exposure in exposure_by_class.items():
            if asset_class in self.asset_class_limits:
                max_allocation = self.asset_class_limits[asset_class].get("max_allocation", 100)
                allocation_pct = (exposure / account_value) * 100 if account_value > 0 else 0
                usage_pct = (allocation_pct / max_allocation) * 100 if max_allocation > 0 else 0
                
                heat_map[asset_class] = {
                    "exposure": exposure,
                    "allocation_pct": allocation_pct,
                    "max_allocation": max_allocation,
                    "usage_pct": usage_pct,
                    "risk_level": self._get_risk_level(usage_pct)
                }
        
        # Global portfolio risk calculations
        leverage = self.portfolio_metrics.get("leverage", 0)
        max_leverage = self.global_risk_limits.get("max_leverage", 2.0)
        leverage_usage = (leverage / max_leverage) * 100 if max_leverage > 0 else 0
        
        heat_map["portfolio"] = {
            "leverage": leverage,
            "max_leverage": max_leverage,
            "leverage_usage": leverage_usage,
            "risk_level": self._get_risk_level(leverage_usage),
            "current_drawdown": self.portfolio_metrics.get("current_drawdown_pct", 0),
            "max_drawdown": self.portfolio_metrics.get("max_drawdown_pct", 0),
            "drawdown_severity": (self.portfolio_metrics.get("current_drawdown_pct", 0) / 
                                 self.global_risk_limits.get("max_drawdown", 15)) * 100
        }
        
        # Store the heat map
        self.portfolio_metrics["heat_map"] = heat_map
    
    def _get_risk_level(self, usage_percentage: float) -> str:
        """
        Convert usage percentage to risk level.
        
        Args:
            usage_percentage: Percentage of limit used
            
        Returns:
            Risk level string
        """
        if usage_percentage < 50:
            return RiskLevel.LOW
        elif usage_percentage < 75:
            return RiskLevel.MODERATE
        elif usage_percentage < 90:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME
    
    def check_trade_risk(self, 
                         symbol: str, 
                         direction: str, 
                         quantity: Union[int, float],
                         entry_price: float, 
                         stop_price: float,
                         target_price: Optional[float] = None,
                         asset_class: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if a potential trade meets risk parameters.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction ("long" or "short")
            quantity: Proposed position size
            entry_price: Proposed entry price
            stop_price: Proposed stop loss price
            target_price: Proposed target price (optional)
            asset_class: Asset class override (optional)
            
        Returns:
            Dictionary with risk assessment
        """
        # Default result structure
        result = {
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "risk_checks_passed": True,
            "warnings": [],
            "errors": [],
            "risk_details": {}
        }
        
        # Check for anomaly cooldown first
        if self.anomaly_detection_enabled and self.anomaly_risk_handler is not None:
            is_cooldown, cooldown_end = self.is_in_anomaly_cooldown()
            if is_cooldown:
                # If in cooldown, check if trading is allowed
                risk_status = self.get_anomaly_risk_status()
                if risk_status.get("trading_restricted", False):
                    time_remaining = cooldown_end - datetime.now() if cooldown_end else timedelta(minutes=10)
                    minutes_remaining = time_remaining.total_seconds() / 60 if time_remaining else 10
                    
                    result["errors"].append(
                        f"Trading restricted due to anomaly cooldown for {minutes_remaining:.1f} more minutes"
                    )
                    result["risk_checks_passed"] = False
                    return result
                else:
                    # Trading allowed but with warnings
                    result["warnings"].append("Trading during anomaly cooldown period - risk adjustments applied")

            # Continue with regular risk checks

            # Apply stop loss modifier if applicable
            stop_loss_modifier = self.active_risk_modifiers.get("stop_loss_modifier", 1.0)
            if stop_loss_modifier < 1.0:
                # Make stop loss tighter
                if direction == "long":
                    adjusted_stop = entry_price - ((entry_price - stop_price) * stop_loss_modifier)
                else:  # short
                    adjusted_stop = stop_price + ((stop_price - entry_price) * stop_loss_modifier)
                
                # Add warning about adjusted stop loss
                result["warnings"].append(
                    f"Stop loss tightened due to market anomalies: {stop_price:.4f} → {adjusted_stop:.4f}"
                )
                
                # Update the stop price for subsequent risk checks
                stop_price = adjusted_stop
                result["stop_price"] = adjusted_stop
        
        try:
            # Detect asset class if not provided
            if asset_class is None:
                asset_class = self._detect_asset_class(symbol)
            result["asset_class"] = asset_class
            
            # Calculate risk amount and percentage
            account_size = self.adapter.get_account_balance()
            
            # Calculate dollar risk
            if direction == "long":
                risk_per_unit = entry_price - stop_price
            else:  # short
                risk_per_unit = stop_price - entry_price
                
            # Ensure risk is positive
            risk_per_unit = abs(risk_per_unit)
            
            # Calculate total dollar risk
            dollar_risk = risk_per_unit * quantity
            
            # Calculate risk as percentage of account
            risk_percentage = (dollar_risk / account_size) * 100
            
            # Calculate reward if target is provided
            reward_risk_ratio = None
            if target_price is not None:
                if direction == "long":
                    reward_per_unit = target_price - entry_price
                else:  # short
                    reward_per_unit = entry_price - target_price
                    
                reward_per_unit = abs(reward_per_unit)
                dollar_reward = reward_per_unit * quantity
                
                # Calculate reward:risk ratio
                if dollar_risk > 0:
                    reward_risk_ratio = dollar_reward / dollar_risk
            
            # Store risk details
            result["risk_details"] = {
                "account_size": account_size,
                "dollar_risk": dollar_risk,
                "risk_percentage": risk_percentage,
                "reward_risk_ratio": reward_risk_ratio,
                "asset_class": asset_class
            }
            
            # Perform risk checks
            self._check_max_risk_per_trade(result)
            self._check_daily_risk_limit(result)
            self._check_asset_class_limits(result)
            self._check_max_open_positions(result)
            self._check_correlation_risk(result)
            self._check_portfolio_constraints(result)
            self._check_reward_risk_ratio(result)
            
            # Check if any errors were found
            result["risk_checks_passed"] = len(result["errors"]) == 0
            
        except Exception as e:
            logger.error(f"Error in trade risk check: {e}")
            result["errors"].append(f"Risk check error: {str(e)}")
            result["risk_checks_passed"] = False
        
        return result
    
    def _detect_asset_class(self, symbol: str) -> str:
        """
        Detect asset class from symbol format if not explicitly provided.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Detected asset class
        """
        # Check forex pairs
        if "/" in symbol and len(symbol.split("/")[0]) == 3 and len(symbol.split("/")[1]) == 3:
            return "forex"
            
        # Check crypto format
        if "-" in symbol and symbol.split("-")[1] in ["USD", "USDT", "BTC"]:
            return "crypto"
            
        # Check futures format
        if "=" in symbol and symbol.endswith("=F"):
            return "futures"
            
        # Check options format
        if "_" in symbol and (symbol.endswith("C") or symbol.endswith("P")):
            return "options"
            
        # Default to equity
        return "equity"
    
    def _check_max_risk_per_trade(self, result: Dict[str, Any]) -> None:
        """Check if trade exceeds maximum risk per trade limit."""
        risk_percentage = result["risk_details"]["risk_percentage"]
        max_risk = self.risk_profile.get("max_risk_per_trade", 1.0)
        
        if risk_percentage > max_risk:
            result["errors"].append(
                f"Trade risk ({risk_percentage:.2f}%) exceeds maximum allowed ({max_risk:.2f}%)"
            )
    
    def _check_daily_risk_limit(self, result: Dict[str, Any]) -> None:
        """Check if trade would exceed daily risk limits."""
        # Get today's date
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Initialize daily risk tracking if needed
        if today not in self.daily_risk_usage:
            self.daily_risk_usage[today] = {
                "total_risk": 0.0,
                "trades": []
            }
        
        # Add current trade risk to today's usage
        risk_percentage = result["risk_details"]["risk_percentage"]
        projected_daily_risk = self.daily_risk_usage[today]["total_risk"] + risk_percentage
        
        # Check against limit
        max_daily_risk = self.risk_profile.get("max_daily_risk", 3.0)
        
        if projected_daily_risk > max_daily_risk:
            result["errors"].append(
                f"Trade would exceed daily risk limit: {projected_daily_risk:.2f}% vs {max_daily_risk:.2f}%"
            )
    
    def _check_asset_class_limits(self, result: Dict[str, Any], asset_class: str) -> None:
        """Check if trade would exceed asset class allocation limits."""
        # Skip if unknown asset class or not in limits
        if asset_class == "unknown" or asset_class not in self.asset_class_limits:
            return
            
        # Get account value
        account_value = result["risk_details"]["account_size"]
        
        # Calculate position value
        quantity = result["quantity"]
        entry_price = result["entry_price"]
        position_value = quantity * entry_price
        
        # Get current allocation to this asset class
        current_allocation = self.portfolio_metrics.get("exposure_by_class", {}).get(asset_class, 0)
        projected_allocation = current_allocation + position_value
        projected_allocation_pct = (projected_allocation / account_value) * 100 if account_value > 0 else 0
        
        # Check against max allocation
        max_allocation = self.asset_class_limits[asset_class].get("max_allocation", 100)
        
        if projected_allocation_pct > max_allocation:
            result["errors"].append(
                f"Trade would exceed {asset_class} allocation limit: {projected_allocation_pct:.2f}% vs {max_allocation:.2f}%"
            )
        
        # Check single position limit
        position_allocation_pct = (position_value / account_value) * 100 if account_value > 0 else 0
        max_single_position = self.asset_class_limits[asset_class].get("max_single_position", 10)
        
        if position_allocation_pct > max_single_position:
            result["errors"].append(
                f"Position size exceeds single position limit: {position_allocation_pct:.2f}% vs {max_single_position:.2f}%"
            )
    
    def _check_max_open_positions(self, result: Dict[str, Any]) -> None:
        """Check if trade would exceed maximum open positions."""
        open_positions = len(self.open_trades)
        max_positions = self.global_risk_limits.get("max_open_positions", 20)
        
        if open_positions >= max_positions:
            result["errors"].append(
                f"Maximum open positions reached: {open_positions} vs {max_positions} limit"
            )
    
    def _check_correlation_risk(self, result: Dict[str, Any], symbol: str) -> None:
        """Check if trade would create excessive correlation risk."""
        # Skip if we don't have correlation data
        if "correlation_matrix" not in self.portfolio_metrics or self.portfolio_metrics["correlation_matrix"].empty:
            return
            
        # Skip if symbol isn't in correlation matrix
        corr_matrix = self.portfolio_metrics["correlation_matrix"]
        if symbol not in corr_matrix.columns:
            return
            
        # Get correlation threshold
        threshold = self.global_risk_limits.get("correlation_threshold", 0.7)
        max_correlated = self.global_risk_limits.get("max_correlated_positions", 3)
        
        # Find highly correlated existing positions
        correlated_positions = []
        for other_symbol in corr_matrix.columns:
            if other_symbol == symbol:
                continue
                
            # Check if we have an open position in the other symbol
            if other_symbol in self.open_trades and abs(corr_matrix.loc[symbol, other_symbol]) >= threshold:
                correlated_positions.append({
                    "symbol": other_symbol,
                    "correlation": corr_matrix.loc[symbol, other_symbol]
                })
        
        # Add to result
        result["risk_details"]["correlated_positions"] = correlated_positions
        
        # Check limit
        if len(correlated_positions) >= max_correlated:
            result["warnings"].append(
                f"Trade has high correlation with {len(correlated_positions)} existing positions"
            )
    
    def _check_portfolio_constraints(self, result: Dict[str, Any]) -> None:
        """Check overall portfolio constraints."""
        # Check drawdown limits
        current_drawdown = self.portfolio_metrics.get("current_drawdown_pct", 0)
        max_drawdown = self.global_risk_limits.get("max_drawdown", 15.0)
        severe_drawdown = self.global_risk_limits.get("severe_drawdown", 25.0)
        
        if current_drawdown >= severe_drawdown:
            result["errors"].append(
                f"Account in severe drawdown ({current_drawdown:.2f}%), trading should be stopped"
            )
        elif current_drawdown >= max_drawdown:
            result["warnings"].append(
                f"Account in significant drawdown ({current_drawdown:.2f}%), consider reducing risk"
            )
        
        # Check leverage limits
        current_leverage = self.portfolio_metrics.get("leverage", 0)
        max_leverage = self.global_risk_limits.get("max_leverage", 2.0)
        
        if current_leverage >= max_leverage:
            result["errors"].append(
                f"Account at maximum leverage: {current_leverage:.2f}x vs {max_leverage:.2f}x limit"
            )
        
        # Check maximum positions
        open_positions = len(self.open_trades)
        max_positions = self.global_risk_limits.get("max_open_positions", 20)
        
        if open_positions >= max_positions:
            result["errors"].append(
                f"Maximum open positions reached: {open_positions} vs {max_positions} limit"
            )
    
    def _check_reward_risk_ratio(self, result: Dict[str, Any]) -> None:
        """Check reward:risk ratio requirement."""
        # Skip if no target price was provided
        if result["target_price"] is None:
            return
            
        # Get reward:risk ratio
        reward_risk_ratio = result["risk_details"]["reward_risk_ratio"]
        min_ratio = self.global_risk_limits.get("min_reward_risk_ratio", 1.5)
        
        if reward_risk_ratio < min_ratio:
            result["warnings"].append(
                f"Low reward:risk ratio: {reward_risk_ratio:.2f} vs {min_ratio:.2f} minimum"
            )
    
    def calculate_position_size(self, 
                              symbol: str, 
                              entry_price: float, 
                              stop_price: float,
                              target_price: Optional[float] = None,
                              risk_percentage: Optional[float] = None,
                              asset_class: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_price: Planned stop loss price
            target_price: Planned target price (optional)
            risk_percentage: Risk percentage override (optional)
            asset_class: Asset class override (optional)
            
        Returns:
            Dictionary with position sizing information
        """
        try:
            # Get account info
            account_info = self.adapter.get_account_info()
            account_value = account_info.get("equity", account_info.get("balance", 0))
            
            # Use default risk percentage if not provided
            if risk_percentage is None:
                risk_percentage = self.risk_profile.get("max_risk_per_trade", 1.0)
                
                # Apply market regime adjustment if applicable
                current_regime = self.risk_profile.get("current_regime", "neutral")
                if current_regime == "bearish":
                    risk_percentage *= 0.7  # Reduce risk in bearish regimes
                elif current_regime == "bullish":
                    risk_percentage *= 1.1  # Can slightly increase risk in bullish regimes
                
                # Apply drawdown-based adjustment
                current_drawdown = self.portfolio_metrics.get("current_drawdown_pct", 0)
                if current_drawdown > 0:
                    max_drawdown = self.global_risk_limits.get("max_drawdown", 15.0)
                    if current_drawdown > (max_drawdown * 0.7):
                        # Close to max drawdown, reduce risk
                        risk_adjustment = 1 - (current_drawdown / max_drawdown)
                        risk_percentage *= max(0.5, risk_adjustment)
            
            # Calculate position size using the adapter
            if asset_class:
                # If asset class provided, use specific calculation for that class
                if asset_class == "futures":
                    sizing = self.adapter._calculate_futures_position(
                        symbol, entry_price, stop_price, account_value, risk_percentage
                    )
                elif asset_class == "forex":
                    sizing = self.adapter._calculate_forex_position(
                        symbol, entry_price, stop_price, account_value, risk_percentage
                    )
                elif asset_class == "crypto":
                    sizing = self.adapter._calculate_crypto_position(
                        symbol, entry_price, stop_price, account_value, risk_percentage
                    )
                else:
                    # Default to adapter's general calculate_position_size
                    sizing = self.adapter.calculate_position_size(
                        symbol, entry_price, stop_price, account_value, risk_percentage
                    )
            else:
                # Use adapter's calculate_position_size which detects asset class
                sizing = self.adapter.calculate_position_size(
                    symbol, entry_price, stop_price, account_value, risk_percentage
                )
            
            # Apply additional risk management constraints
            position_size_multiplier = self.risk_profile.get("position_size_multiplier", 1.0)
            if position_size_multiplier != 1.0:
                # Adjust position size by the multiplier
                sizing["original_position_size"] = sizing.get("position_size", 0)
                sizing["position_size"] *= position_size_multiplier
                
                # Update other related fields
                if "contracts" in sizing:
                    sizing["contracts"] = int(sizing["contracts"] * position_size_multiplier)
                if "units" in sizing:
                    sizing["units"] *= position_size_multiplier
                if "lots" in sizing:
                    sizing["lots"] *= position_size_multiplier
                
                sizing["position_size_multiplier"] = position_size_multiplier
            
            # Add reward:risk ratio if target price provided
            if target_price is not None:
                risk_per_unit = abs(entry_price - stop_price)
                reward_per_unit = abs(target_price - entry_price)
                
                reward_risk_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0
                reward_amount = reward_per_unit * sizing["position_size"]
                
                sizing["target_price"] = target_price
                sizing["reward_per_unit"] = reward_per_unit
                sizing["reward_amount"] = reward_amount
                sizing["reward_risk_ratio"] = reward_risk_ratio
            
            # Apply anomaly risk modifier to the calculated position size
            if self.anomaly_detection_enabled and self.anomaly_risk_handler is not None:
                if hasattr(self, 'result'):  # If result is defined in the original implementation
                    # Apply position size modifier from anomaly risk handler
                    position_size_modifier = self.active_risk_modifiers.get("position_size_modifier", 1.0)
                    
                    # Adjust the calculated position size
                    if "quantity" in self.result:
                        original_quantity = self.result["quantity"]
                        adjusted_quantity = original_quantity * position_size_modifier
                        
                        # Round to appropriate precision
                        if isinstance(original_quantity, int):
                            adjusted_quantity = int(adjusted_quantity)
                        else:
                            # Round to same number of decimal places
                            adjusted_quantity = round(adjusted_quantity, 8)
                        
                        self.result["quantity"] = adjusted_quantity
                        self.result["adjusted_for_anomaly"] = True
                        self.result["position_size_modifier"] = position_size_modifier
                        
                        if position_size_modifier < 1.0:
                            self.result["notes"].append(
                                f"Position size reduced by {(1-position_size_modifier)*100:.0f}% due to market anomalies"
                            )
            
            return sizing
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {
                "position_size": 0,
                "error": str(e),
                "risk_percentage": risk_percentage
            }
    
    def record_trade_entry(self, 
                          symbol: str, 
                          direction: str,
                          quantity: Union[int, float],
                          entry_price: float,
                          stop_price: float,
                          target_price: Optional[float] = None,
                          asset_class: Optional[str] = None,
                          strategy: Optional[str] = None,
                          trade_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Record a new trade entry for risk tracking.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction ("long" or "short")
            quantity: Position size
            entry_price: Entry price
            stop_price: Stop loss price
            target_price: Target price (optional)
            asset_class: Asset class (optional)
            strategy: Trading strategy (optional)
            trade_id: Unique trade ID (optional)
            
        Returns:
            Dictionary with trade record
        """
        # Generate trade ID if not provided
        if not trade_id:
            trade_id = f"{symbol}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
        # Create trade record
        trade = {
            "trade_id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "entry_time": datetime.now().isoformat(),
            "asset_class": asset_class or self._detect_asset_class(symbol),
            "strategy": strategy,
            "risk_profile": {
                "account_value": self.portfolio_metrics.get("account_value", 0),
                "risk_percentage": self.risk_profile.get("max_risk_per_trade", 1.0),
                "market_regime": self.risk_profile.get("current_regime", "neutral")
            }
        }
        
        # Calculate risk metrics
        risk_per_unit = abs(entry_price - stop_price)
        risk_amount = risk_per_unit * quantity
        risk_percentage = (risk_amount / trade["risk_profile"]["account_value"]) * 100 if trade["risk_profile"]["account_value"] > 0 else 0
        
        trade["risk_per_unit"] = risk_per_unit
        trade["risk_amount"] = risk_amount
        trade["risk_percentage"] = risk_percentage
        
        # Calculate reward metrics if target provided
        if target_price is not None:
            reward_per_unit = abs(target_price - entry_price)
            reward_amount = reward_per_unit * quantity
            reward_risk_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0
            
            trade["reward_per_unit"] = reward_per_unit
            trade["reward_amount"] = reward_amount
            trade["reward_risk_ratio"] = reward_risk_ratio
        
        # Store trade in open trades
        self.open_trades[symbol] = trade
        
        # Save open trades
        self._save_open_trades()
        
        # Update daily risk tracking
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.daily_risk_usage:
            self.daily_risk_usage[today] = {
                "total_risk": 0.0,
                "trades": []
            }
        
        self.daily_risk_usage[today]["total_risk"] += risk_percentage
        self.daily_risk_usage[today]["trades"].append({
            "trade_id": trade_id,
            "symbol": symbol,
            "risk_percentage": risk_percentage,
            "time": datetime.now().isoformat()
        })
        
        # Update portfolio metrics
        self.update_portfolio_metrics()
        
        return trade
    
    def record_trade_exit(self, 
                         trade_id: Optional[str] = None,
                         symbol: Optional[str] = None,
                         exit_price: float = 0,
                         exit_time: Optional[str] = None,
                         pnl: Optional[float] = None,
                         exit_reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Record a trade exit for risk tracking.
        
        Args:
            trade_id: Trade ID to close (optional if symbol provided)
            symbol: Symbol to close (optional if trade_id provided)
            exit_price: Exit price
            exit_time: Exit time (optional, defaults to now)
            pnl: Profit/loss amount (optional)
            exit_reason: Reason for exit (optional)
            
        Returns:
            Dictionary with trade exit record
        """
        # Find the trade to close
        trade = None
        
        if trade_id:
            # Find by trade ID
            for symbol_key, trade_record in self.open_trades.items():
                if trade_record.get("trade_id") == trade_id:
                    trade = trade_record
                    symbol = symbol_key
                    break
        elif symbol:
            # Find by symbol
            if symbol in self.open_trades:
                trade = self.open_trades[symbol]
                trade_id = trade.get("trade_id")
        
        if not trade:
            logger.warning(f"No open trade found for trade_id={trade_id}, symbol={symbol}")
            return {"error": "No open trade found"}
        
        # Set exit time if not provided
        if not exit_time:
            exit_time = datetime.now().isoformat()
            
        # Calculate PnL if not provided
        if pnl is None and exit_price > 0:
            entry_price = trade.get("entry_price", 0)
            quantity = trade.get("quantity", 0)
            direction = trade.get("direction", "long")
            
            if direction == "long":
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity
        
        # Set exit details
        trade["exit_price"] = exit_price
        trade["exit_time"] = exit_time
        trade["pnl"] = pnl
        trade["exit_reason"] = exit_reason
        trade["trade_duration"] = self._calculate_trade_duration(trade.get("entry_time"), exit_time)
        
        # Calculate PnL percentage
        if pnl is not None:
            risk_amount = trade.get("risk_amount", 0)
            if risk_amount > 0:
                trade["pnl_risk_ratio"] = pnl / risk_amount
            
            # Calculate with reference to the entry value
            entry_value = trade.get("entry_price", 0) * trade.get("quantity", 0)
            if entry_value > 0:
                trade["pnl_percentage"] = (pnl / entry_value) * 100
        
        # Record in risk history
        if trade.get("trade_id"):
            self.risk_history.append(trade.copy())
            self._save_risk_history()
        
        # Remove from open trades
        if symbol in self.open_trades:
            del self.open_trades[symbol]
            self._save_open_trades()
        
        # Update portfolio metrics
        self.update_portfolio_metrics()
        
        return trade
    
    def _calculate_trade_duration(self, entry_time: Optional[str], exit_time: Optional[str]) -> Optional[int]:
        """Calculate trade duration in minutes."""
        if not entry_time or not exit_time:
            return None
            
        try:
            entry = datetime.fromisoformat(entry_time)
            exit = datetime.fromisoformat(exit_time)
            duration = exit - entry
            return int(duration.total_seconds() / 60)
        except:
            return None
    
    def _save_risk_history(self) -> None:
        """Save risk history to journal."""
        journal_file = os.path.join(self.journal_dir, "risk_history.json")
        try:
            with open(journal_file, 'w') as f:
                json.dump(self.risk_history[-1000:], f, indent=2)  # Keep only last 1000 trades
        except Exception as e:
            logger.error(f"Failed to save risk history: {e}")
    
    def update_risk_profile(self, profile_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the risk profile settings.
        
        Args:
            profile_updates: Dictionary with profile settings to update
            
        Returns:
            Updated risk profile
        """
        for key, value in profile_updates.items():
            if key in self.risk_profile:
                self.risk_profile[key] = value
        
        # Save updated profile to config
        self._save_risk_config()
        
        return self.risk_profile
    
    def update_global_limits(self, limit_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the global risk limits.
        
        Args:
            limit_updates: Dictionary with limit settings to update
            
        Returns:
            Updated global limits
        """
        for key, value in limit_updates.items():
            if key in self.global_risk_limits:
                self.global_risk_limits[key] = value
        
        # Save updated limits to config
        self._save_risk_config()
        
        return self.global_risk_limits
    
    def _save_risk_config(self) -> None:
        """Save current risk configuration."""
        config_file = os.path.join(self.journal_dir, "risk_config.json")
        config = {
            "risk_profile": self.risk_profile,
            "global_limits": self.global_risk_limits,
            "asset_class_limits": self.asset_class_limits
        }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save risk config: {e}")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive risk report.
        
        Returns:
            Dictionary with risk report data
        """
        # Update portfolio metrics
        self.update_portfolio_metrics()
        
        # Generate report
        report = {
            "portfolio_metrics": self.portfolio_metrics,
            "risk_profile": self.risk_profile,
            "global_limits": self.global_risk_limits,
            "open_trades": len(self.open_trades),
            "open_trade_details": list(self.open_trades.values()),
            "risk_heat_map": self.portfolio_metrics.get("heat_map", {}),
            "report_time": datetime.now().isoformat()
        }
        
        # Add daily risk usage
        today = datetime.now().strftime("%Y-%m-%d")
        report["daily_risk_usage"] = self.daily_risk_usage.get(today, {"total_risk": 0.0, "trades": []})
        
        # Add correlated positions warnings
        if "highly_correlated_pairs" in self.portfolio_metrics:
            report["correlation_warnings"] = self.portfolio_metrics["highly_correlated_pairs"]
        
        # Risk capacity calculation
        max_daily_risk = self.risk_profile.get("max_daily_risk", 3.0)
        used_daily_risk = report["daily_risk_usage"]["total_risk"]
        report["remaining_risk_capacity"] = max(0, max_daily_risk - used_daily_risk)
        
        # Risk distribution by asset class
        report["risk_distribution"] = {}
        for asset_class in self.asset_class_limits:
            allocation_pct = 0
            if asset_class in self.portfolio_metrics.get("exposure_by_class", {}):
                exposure = self.portfolio_metrics["exposure_by_class"][asset_class]
                account_value = self.portfolio_metrics.get("account_value", 0)
                if account_value > 0:
                    allocation_pct = (exposure / account_value) * 100
                    
            max_allocation = self.asset_class_limits[asset_class].get("max_allocation", 100)
            available_allocation = max(0, max_allocation - allocation_pct)
            
            report["risk_distribution"][asset_class] = {
                "current_allocation": allocation_pct,
                "max_allocation": max_allocation,
                "available_allocation": available_allocation,
                "utilization_percentage": (allocation_pct / max_allocation) * 100 if max_allocation > 0 else 0
            }
        
        return report
    
    def update_risk_from_anomalies(self, anomaly_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update risk parameters based on detected market anomalies.
        
        Args:
            anomaly_result: Dictionary with anomaly detection results
            
        Returns:
            Dictionary with risk adjustment actions taken
        """
        if not self.anomaly_detection_enabled or self.anomaly_risk_handler is None:
            logger.debug("Anomaly detection is disabled, skipping risk update")
            return {"enabled": False}
        
        # Process the anomaly result through the risk handler
        risk_actions = self.anomaly_risk_handler.process_anomaly_result(anomaly_result)
        
        # Update risk modifiers
        if "position_size_modifier" in risk_actions:
            self.active_risk_modifiers["position_size_modifier"] = risk_actions["position_size_modifier"]
        
        if "stop_loss_modifier" in risk_actions:
            self.active_risk_modifiers["stop_loss_modifier"] = risk_actions["stop_loss_modifier"]
        
        # Check if we need to enter cooldown mode
        is_in_cooldown, cooldown_end, cooldown_level = self.anomaly_risk_handler.get_active_cooldown()
        self.in_anomaly_cooldown = is_in_cooldown
        self.anomaly_cooldown_end = cooldown_end
        
        # Update risk profile based on anomaly risk level
        if "risk_mode" in risk_actions:
            self.set_risk_mode(risk_actions["risk_mode"])
        
        # Log the risk adjustments
        logger.info(f"Risk adjusted for anomalies: {risk_actions}")
        if is_in_cooldown:
            logger.warning(
                f"Anomaly cooldown active until {cooldown_end} due to {cooldown_level} risk level"
            )
        
        return risk_actions
    
    def set_risk_mode(self, mode: str) -> None:
        """
        Set the risk mode based on anomaly or other risk factors.
        
        Args:
            mode: Risk mode identifier
        """
        if mode == "normal":
            # Reset to default risk profile
            self.risk_profile["position_size_multiplier"] = 1.0
            logger.info("Risk mode set to NORMAL")
            
        elif mode == "cautious":
            # More cautious trading
            self.risk_profile["position_size_multiplier"] = 0.7
            self.risk_profile["max_risk_per_trade"] = self.risk_profile.get("max_risk_per_trade", 1.0) * 0.7
            logger.info("Risk mode set to CAUTIOUS")
            
        elif mode == "defensive":
            # Defensive trading
            self.risk_profile["position_size_multiplier"] = 0.4
            self.risk_profile["max_risk_per_trade"] = self.risk_profile.get("max_risk_per_trade", 1.0) * 0.5
            logger.info("Risk mode set to DEFENSIVE")
            
        elif mode == "lockdown":
            # Lockdown mode - minimal trading
            self.risk_profile["position_size_multiplier"] = 0.0  # No new positions
            logger.warning("Risk mode set to LOCKDOWN - new positions blocked")
        
        else:
            logger.warning(f"Unknown risk mode: {mode}, ignoring")
    
    def is_in_anomaly_cooldown(self) -> Tuple[bool, Optional[datetime]]:
        """
        Check if the risk manager is currently in an anomaly-triggered cooldown period.
        
        Returns:
            Tuple of (is_in_cooldown, cooldown_end_time)
        """
        if not self.anomaly_detection_enabled or self.anomaly_risk_handler is None:
            return False, None
        
        # If the last check was recent, use cached value
        if self.in_anomaly_cooldown and self.anomaly_cooldown_end:
            if datetime.now() < self.anomaly_cooldown_end:
                return True, self.anomaly_cooldown_end
        
        # Otherwise, check with the risk handler
        is_in_cooldown, cooldown_end, _ = self.anomaly_risk_handler.get_active_cooldown()
        self.in_anomaly_cooldown = is_in_cooldown
        self.anomaly_cooldown_end = cooldown_end
        
        return is_in_cooldown, cooldown_end
    
    def get_anomaly_risk_status(self) -> Dict[str, Any]:
        """
        Get current anomaly risk status information.
        
        Returns:
            Dictionary with anomaly risk status details
        """
        if not self.anomaly_detection_enabled or self.anomaly_risk_handler is None:
            return {"enabled": False}
        
        # Get status from the risk handler
        return self.anomaly_risk_handler.get_current_risk_status() 