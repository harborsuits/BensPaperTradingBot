#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Management Engine

This module implements an enhanced risk management engine that:
1. Tracks position-level risk allocations
2. Monitors portfolio-level exposure
3. Detects correlation risks between assets
4. Manages drawdown protection
5. Attributes performance to risk factors

The engine integrates with the event bus system to provide full transparency
into all risk management decisions.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import json

from trading_bot.core.event_bus import EventBus, get_global_event_bus, Event
from trading_bot.core.constants import EventType
from trading_bot.data.persistence import PersistenceManager
from trading_bot.risk.risk_manager import RiskManager, RiskLevel, StopLossType
from trading_bot.risk.risk_monitor import RiskMonitor

logger = logging.getLogger(__name__)

class RiskManagementEngine:
    """
    Enhanced risk management engine that provides transparency into all risk decisions.
    
    This class extends the core RiskManager and RiskMonitor capabilities with an
    event-driven architecture that publishes all risk management decisions and insights
    to the central event bus.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 persistence_manager: Optional[PersistenceManager] = None):
        """
        Initialize the risk management engine.
        
        Args:
            config: Configuration dictionary
            persistence_manager: Persistence manager for storing risk data
        """
        self.config = config or {}
        self.persistence = persistence_manager
        self.event_bus = get_global_event_bus()
        
        # Initialize risk parameters
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.05)  # 5% default
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.max_position_size = self.config.get('max_position_size', 0.2)  # 20% default
        self.drawdown_threshold = self.config.get('drawdown_threshold', 0.1)  # 10% default
        self.risk_per_trade = self.config.get('risk_per_trade', 0.01)  # 1% default
        
        # Create the underlying risk manager and monitor
        risk_config = self.config.get('risk_manager', {})
        self.risk_manager = RiskManager(config=risk_config)
        
        monitor_config = self.config.get('risk_monitor', {})
        self.risk_monitor = RiskMonitor(config=monitor_config)
        
        # Internal tracking
        self.positions = {}
        self.correlations = {}
        self.portfolio_value = self.config.get('initial_portfolio_value', 100000.0)
        self.risk_allocations = {}
        self.drawdown_history = []
        
        # Market regime awareness for risk adjustments
        self.current_market_regime = "normal"
        
        logger.info("Risk Management Engine initialized")
    
    def register_event_handlers(self):
        """Register for relevant events from the event bus."""
        # Subscribe to market regime changes to adapt risk parameters
        self.event_bus.subscribe(EventType.MARKET_REGIME_CHANGED, self._on_market_regime_change)
        
        # Subscribe to position updates
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, self._on_trade_executed)
        self.event_bus.subscribe(EventType.TRADE_CLOSED, self._on_trade_closed)
        
        # Subscribe to correlation updates
        self.event_bus.subscribe(EventType.CORRELATION_MATRIX_UPDATED, self._on_correlation_update)

    def assess_position_risk(self, 
                           symbol: str, 
                           position_size: float, 
                           entry_price: float, 
                           stop_loss_price: float) -> Dict[str, Any]:
        """
        Assess the risk of a specific position and publish the assessment.
        
        Args:
            symbol: Trading symbol
            position_size: Size of the position
            entry_price: Entry price
            stop_loss_price: Stop loss price
            
        Returns:
            Dict containing risk assessment details
        """
        # Calculate risk metrics
        risk_amount = abs(entry_price - stop_loss_price) * position_size
        risk_pct = risk_amount / self.portfolio_value
        
        # Get historical volatility for the symbol if available
        volatility = self._get_symbol_volatility(symbol)
        
        # Determine if position exceeds risk limits
        exceeds_per_trade_limit = risk_pct > self.risk_per_trade
        exceeds_position_limit = risk_pct > self.max_position_size
        
        # Combine into risk score (0-100)
        risk_score = min(100, (risk_pct / self.risk_per_trade) * 50 + (volatility / 0.2) * 50)
        
        # Determine appropriate action if needed
        action = None
        reason = None
        
        if exceeds_per_trade_limit:
            action = "reduce_position"
            reason = "trade_risk_exceeded"
        elif exceeds_position_limit:
            action = "reduce_position"
            reason = "position_limit_exceeded"
        
        # Create risk assessment
        assessment = {
            "symbol": symbol,
            "position_size": position_size,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "risk_amount": risk_amount,
            "risk_pct": risk_pct,
            "volatility": volatility,
            "risk_score": risk_score,
            "action": action,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in risk allocations
        self.risk_allocations[symbol] = assessment
        
        # Publish risk allocation event
        self.event_bus.create_and_publish(
            event_type=EventType.RISK_ALLOCATION_CHANGED,
            data=assessment,
            source="risk_management_engine"
        )
        
        return assessment

    def assess_portfolio_risk(self) -> Dict[str, Any]:
        """
        Assess the overall portfolio risk and publish the assessment.
        
        Returns:
            Dict containing portfolio risk assessment
        """
        # Calculate total portfolio risk
        total_risk = sum(alloc["risk_pct"] for alloc in self.risk_allocations.values())
        
        # Calculate margin utilization if available
        margin_used = sum(alloc.get("margin_used", 0) for alloc in self.risk_allocations.values())
        margin_available = self.portfolio_value * 0.5  # Assuming 2:1 margin
        margin_utilization = margin_used / margin_available if margin_available > 0 else 0
        
        # Calculate exposure by sector/asset class if available
        exposures = {}
        for symbol, alloc in self.risk_allocations.items():
            asset_class = alloc.get("asset_class", "unknown")
            if asset_class not in exposures:
                exposures[asset_class] = 0
            exposures[asset_class] += alloc["risk_pct"]
        
        # Determine if portfolio exceeds risk limits
        exceeds_total_risk = total_risk > self.max_portfolio_risk
        
        # Determine appropriate action if needed
        action = None
        if exceeds_total_risk:
            action = "reduce_exposure"
        
        # Create portfolio assessment
        assessment = {
            "total_risk": total_risk,
            "max_risk": self.max_portfolio_risk,
            "margin_utilization": margin_utilization,
            "exposures": exposures,
            "action": action,
            "timestamp": datetime.now().isoformat()
        }
        
        # Publish portfolio exposure event
        self.event_bus.create_and_publish(
            event_type=EventType.PORTFOLIO_EXPOSURE_UPDATED,
            data=assessment,
            source="risk_management_engine"
        )
        
        return assessment

    def detect_correlation_risks(self) -> List[Dict[str, Any]]:
        """
        Detect high correlations between assets that could increase portfolio risk.
        
        Returns:
            List of correlation risk alerts
        """
        alerts = []
        
        # Check correlations between all pairs of active positions
        symbols = list(self.positions.keys())
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                # Skip if we don't have correlation data for this pair
                pair_key = f"{symbol1}/{symbol2}"
                if pair_key not in self.correlations:
                    continue
                
                correlation = self.correlations[pair_key]
                
                # Alert if correlation exceeds threshold
                if abs(correlation) > self.correlation_threshold:
                    alert = {
                        "symbols": [symbol1, symbol2],
                        "correlation": correlation,
                        "threshold": self.correlation_threshold,
                        "action": "diversify_assets",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    alerts.append(alert)
                    
                    # Publish correlation risk alert
                    self.event_bus.create_and_publish(
                        event_type=EventType.CORRELATION_RISK_ALERT,
                        data=alert,
                        source="risk_management_engine"
                    )
        
        return alerts

    def monitor_drawdown(self, current_portfolio_value: float) -> Dict[str, Any]:
        """
        Monitor and react to drawdowns exceeding threshold.
        
        Args:
            current_portfolio_value: Current portfolio value
            
        Returns:
            Dict containing drawdown assessment and actions
        """
        # Update portfolio value
        prev_portfolio_value = self.portfolio_value
        self.portfolio_value = current_portfolio_value
        
        # Get peak portfolio value from history
        peak_value = max([value for _, value in self.drawdown_history]) if self.drawdown_history else current_portfolio_value
        
        # Calculate current drawdown
        current_drawdown = (peak_value - current_portfolio_value) / peak_value if peak_value > 0 else 0
        
        # Add to drawdown history
        self.drawdown_history.append((datetime.now(), current_portfolio_value))
        
        # Prune history if needed
        if len(self.drawdown_history) > 100:
            self.drawdown_history = self.drawdown_history[-100:]
        
        # Check if drawdown exceeds threshold
        exceeded = current_drawdown > self.drawdown_threshold
        
        # Calculate severity level based on multiples of threshold
        severity = min(int(current_drawdown / self.drawdown_threshold), 3) if exceeded else 0
        
        # Determine action based on severity
        action = self._get_drawdown_action(severity)
        
        # Create drawdown assessment
        assessment = {
            "current_drawdown": current_drawdown,
            "threshold": self.drawdown_threshold,
            "exceeded": exceeded,
            "severity": severity,
            "action": action,
            "peak_value": peak_value,
            "current_value": current_portfolio_value,
            "daily_change_pct": (current_portfolio_value / prev_portfolio_value - 1) * 100,
            "timestamp": datetime.now().isoformat()
        }
        
        # Publish drawdown event if threshold exceeded
        if exceeded:
            self.event_bus.create_and_publish(
                event_type=EventType.DRAWDOWN_THRESHOLD_EXCEEDED,
                data=assessment,
                source="risk_management_engine"
            )
        
        return assessment

    def perform_risk_attribution(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attribute performance to various risk factors.
        
        Args:
            performance_data: Performance data for attribution
            
        Returns:
            Dict containing risk attribution results
        """
        # Extract performance data
        total_return = performance_data.get("total_return", 0)
        positions = performance_data.get("positions", {})
        
        # Define risk factors to attribute performance to
        risk_factors = {
            "market_beta": 0.0,
            "sector_exposure": 0.0,
            "volatility_exposure": 0.0,
            "correlation_impact": 0.0,
            "liquidity_risk": 0.0,
            "specific_risk": 0.0  # Idiosyncratic risk
        }
        
        # Calculate attribution (simplified example)
        # In a real implementation, this would use more sophisticated factor models
        # such as Barra, Fama-French, or proprietary models
        if positions:
            # Market beta contribution (systematic risk)
            risk_factors["market_beta"] = total_return * 0.4  # Example: 40% of return attributed to market
            
            # Sector exposure
            risk_factors["sector_exposure"] = total_return * 0.15
            
            # Volatility exposure
            risk_factors["volatility_exposure"] = total_return * 0.1
            
            # Correlation impact
            risk_factors["correlation_impact"] = total_return * 0.05
            
            # Liquidity risk
            risk_factors["liquidity_risk"] = total_return * 0.05
            
            # Specific (idiosyncratic) risk
            risk_factors["specific_risk"] = total_return * 0.25
        
        # Calculate contribution of each risk factor
        attribution = {}
        for factor, exposure in risk_factors.items():
            attribution[factor] = exposure
        
        # Create attribution result
        result = {
            "risk_factors": risk_factors,
            "attribution": attribution,
            "total_return": total_return,
            "timestamp": datetime.now().isoformat()
        }
        
        # Publish risk attribution event
        self.event_bus.create_and_publish(
            event_type=EventType.RISK_ATTRIBUTION_CALCULATED,
            data=result,
            source="risk_management_engine"
        )
        
        return result

    def adjust_risk_parameters(self, market_regime: str) -> Dict[str, Any]:
        """
        Dynamically adjust risk parameters based on market regime.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Dict containing adjusted risk parameters
        """
        # Save previous values for comparison
        prev_max_portfolio_risk = self.max_portfolio_risk
        prev_risk_per_trade = self.risk_per_trade
        prev_correlation_threshold = self.correlation_threshold
        
        # Adjust risk parameters based on market regime
        if market_regime == "volatile":
            # Reduce risk in volatile markets
            self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.05) * 0.7
            self.risk_per_trade = self.config.get('risk_per_trade', 0.01) * 0.7
            self.correlation_threshold = self.config.get('correlation_threshold', 0.7) * 0.9
        
        elif market_regime == "trending":
            # Standard risk in trending markets
            self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.05)
            self.risk_per_trade = self.config.get('risk_per_trade', 0.01)
            self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        
        elif market_regime == "low_volatility":
            # Slightly increased risk in low volatility markets
            self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.05) * 1.1
            self.risk_per_trade = self.config.get('risk_per_trade', 0.01) * 1.1
            self.correlation_threshold = self.config.get('correlation_threshold', 0.7) * 1.1
        
        else:  # ranging or normal
            # Default risk parameters
            self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.05)
            self.risk_per_trade = self.config.get('risk_per_trade', 0.01)
            self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        
        # Create adjustment result
        result = {
            "market_regime": market_regime,
            "max_portfolio_risk": self.max_portfolio_risk,
            "risk_per_trade": self.risk_per_trade,
            "correlation_threshold": self.correlation_threshold,
            "changes": {
                "max_portfolio_risk": self.max_portfolio_risk - prev_max_portfolio_risk,
                "risk_per_trade": self.risk_per_trade - prev_risk_per_trade,
                "correlation_threshold": self.correlation_threshold - prev_correlation_threshold
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return result

    def update_positions(self, positions: Dict[str, Any]):
        """
        Update the current positions being managed.
        
        Args:
            positions: Dictionary of positions
        """
        self.positions = positions
        
        # Recalculate risk metrics and trigger events
        for symbol, position in positions.items():
            self.assess_position_risk(
                symbol=symbol,
                position_size=position.get("size", 0),
                entry_price=position.get("entry_price", 0),
                stop_loss_price=position.get("stop_loss_price", 0)
            )
        
        # Assess overall portfolio risk
        self.assess_portfolio_risk()
        
        # Check for correlation risks
        self.detect_correlation_risks()

    def update_correlations(self, correlation_matrix: Dict[str, Dict[str, float]]):
        """
        Update the correlation matrix used for risk calculations.
        
        Args:
            correlation_matrix: Matrix of correlations between symbols
        """
        # Convert the matrix to a flattened dictionary for easier lookup
        for symbol1, correlations in correlation_matrix.items():
            for symbol2, correlation in correlations.items():
                if symbol1 != symbol2:
                    pair_key = f"{symbol1}/{symbol2}"
                    self.correlations[pair_key] = correlation
        
        # Check for correlation risks with updated data
        self.detect_correlation_risks()

    #
    # Private helper methods
    #
    
    def _get_symbol_volatility(self, symbol: str) -> float:
        """
        Get the historical volatility for a symbol.
        
        Args:
            symbol: Symbol to get volatility for
            
        Returns:
            Volatility as a decimal
        """
        # Placeholder - in a real implementation this would use market data
        # to calculate historical volatility
        return 0.2  # Default 20% volatility
    
    def _get_drawdown_action(self, severity: int) -> str:
        """
        Determine appropriate action based on drawdown severity.
        
        Args:
            severity: Severity level (0-3)
            
        Returns:
            Action to take as a string
        """
        actions = {
            0: "monitor",
            1: "reduce_new_positions",
            2: "reduce_overall_exposure",
            3: "emergency_portfolio_protection"
        }
        return actions.get(severity, "monitor")
    
    def _on_market_regime_change(self, event: Event):
        """
        Handle market regime change events.
        
        Args:
            event: Market regime change event
        """
        regime = event.data.get("current_regime")
        if regime:
            self.current_market_regime = regime
            self.adjust_risk_parameters(regime)
    
    def _on_trade_executed(self, event: Event):
        """
        Handle trade executed events.
        
        Args:
            event: Trade executed event
        """
        symbol = event.data.get("symbol")
        size = event.data.get("size")
        price = event.data.get("price")
        stop_loss = event.data.get("stop_loss")
        
        if symbol and size and price:
            # Add to positions
            self.positions[symbol] = {
                "size": size,
                "entry_price": price,
                "stop_loss_price": stop_loss,
                "timestamp": event.data.get("timestamp", datetime.now().isoformat())
            }
            
            # Assess position risk
            self.assess_position_risk(symbol, size, price, stop_loss or price * 0.95)
            
            # Reassess portfolio risk
            self.assess_portfolio_risk()
    
    def _on_trade_closed(self, event: Event):
        """
        Handle trade closed events.
        
        Args:
            event: Trade closed event
        """
        symbol = event.data.get("symbol")
        
        if symbol and symbol in self.positions:
            # Remove from positions
            del self.positions[symbol]
            
            # Remove from risk allocations
            if symbol in self.risk_allocations:
                del self.risk_allocations[symbol]
            
            # Reassess portfolio risk
            self.assess_portfolio_risk()
    
    def _on_correlation_update(self, event: Event):
        """
        Handle correlation matrix update events.
        
        Args:
            event: Correlation matrix update event
        """
        matrix = event.data.get("matrix")
        if matrix:
            self.update_correlations(matrix)
