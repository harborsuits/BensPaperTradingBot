"""
Decision engine for trading opportunities.

This module implements the main decision engine that processes opportunities,
applies policy and compliance rules, and generates trading decisions.
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable

from trading_bot.policy.types import Policy, Instrument, Regime
from trading_bot.policy.service import PolicyService
from trading_bot.engine.router import rank_and_route, Opportunity, RoutedOpportunity
from trading_bot.engine.audit import get_decision_stats


logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Engine for processing trading opportunities and making decisions.
    
    This class integrates policy, compliance, scoring, and routing
    to generate trading decisions from opportunities.
    """
    
    def __init__(self, policy_service: PolicyService):
        """
        Initialize the decision engine.
        
        Args:
            policy_service: Service for managing trading policies
        """
        self.policy_service = policy_service
        self.enabled = True
        self.last_daily_loss_check = 0
        self.daily_loss_pct = 0.0
        self.order_callbacks: List[Callable[[RoutedOpportunity], None]] = []
    
    def process_opportunities(
        self,
        opportunities: List[Dict[str, Any]],
        current_regime: Optional[Regime] = None,
        now: Optional[int] = None
    ) -> List[RoutedOpportunity]:
        """
        Process a batch of trading opportunities.
        
        Args:
            opportunities: List of trading opportunities
            current_regime: Current market regime (optional)
            now: Current timestamp in milliseconds (optional)
            
        Returns:
            List of routed opportunities
        """
        if not self.enabled:
            logger.info("Decision engine is disabled, skipping opportunity processing")
            return []
        
        # Use current time if not provided
        if now is None:
            now = int(time.time() * 1000)
        
        # Check if we need to reload the policy
        self.policy_service.reload_if_changed()
        
        # Get the current policy
        policy = self.policy_service.get_policy()
        
        # Check daily loss limit
        if self._check_daily_loss_limit(policy, now):
            logger.warning("Daily loss limit exceeded, disabling decision engine")
            self.enabled = False
            return []
        
        # Convert opportunities to internal format
        processed_opps = self._preprocess_opportunities(opportunities, current_regime, now)
        
        # Rank and route opportunities
        routed_opps = rank_and_route(processed_opps, policy, now)
        
        # Execute callbacks for routed opportunities
        for opp in routed_opps:
            self._execute_callbacks(opp)
        
        return routed_opps
    
    def register_order_callback(self, callback: Callable[[RoutedOpportunity], None]) -> None:
        """
        Register a callback to be called for each routed opportunity.
        
        Args:
            callback: Function to call with each routed opportunity
        """
        self.order_callbacks.append(callback)
    
    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable the decision engine.
        
        Args:
            enabled: Whether the engine should be enabled
        """
        self.enabled = enabled
        logger.info(f"Decision engine {'enabled' if enabled else 'disabled'}")
    
    def update_daily_loss(self, loss_pct: float) -> None:
        """
        Update the current daily loss percentage.
        
        Args:
            loss_pct: Daily loss as a percentage of portfolio
        """
        self.daily_loss_pct = loss_pct
        self.last_daily_loss_check = int(time.time() * 1000)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the decision engine.
        
        Returns:
            Dictionary with engine status
        """
        policy = self.policy_service.get_policy()
        decision_stats = get_decision_stats()
        
        return {
            "enabled": self.enabled,
            "policy_version": policy["version"],
            "daily_loss_pct": self.daily_loss_pct,
            "daily_loss_limit": policy["risk"]["max_daily_loss_pct"],
            "decision_stats": decision_stats
        }
    
    def _check_daily_loss_limit(self, policy: Policy, now: int) -> bool:
        """
        Check if the daily loss limit has been exceeded.
        
        Args:
            policy: The current policy
            now: Current timestamp in milliseconds
            
        Returns:
            True if the limit has been exceeded, False otherwise
        """
        # Skip if we haven't checked recently
        if now - self.last_daily_loss_check > 60_000:  # 1 minute
            return False
        
        # Check if we've exceeded the daily loss limit
        if abs(self.daily_loss_pct) >= policy["risk"]["max_daily_loss_pct"]:
            return True
        
        return False
    
    def _preprocess_opportunities(
        self,
        opportunities: List[Dict[str, Any]],
        current_regime: Optional[Regime],
        now: int
    ) -> List[Opportunity]:
        """
        Preprocess opportunities into the internal format.
        
        Args:
            opportunities: List of raw opportunities
            current_regime: Current market regime
            now: Current timestamp in milliseconds
            
        Returns:
            List of processed opportunities
        """
        result = []
        
        for opp in opportunities:
            # Skip opportunities without required fields
            if not all(k in opp for k in ["instrument", "symbol", "alpha"]):
                continue
            
            # Generate ID if not provided
            opp_id = opp.get("id", str(uuid.uuid4()))
            
            # Get timestamp if not provided
            ts = opp.get("ts", now)
            
            # Calculate regime alignment if regime is provided
            regime_align = 0.0
            if current_regime and "regime_align" not in opp:
                # Simple alignment logic - can be more sophisticated
                if opp["instrument"] == Instrument.ETF.value:
                    if current_regime == Regime.RISK_ON:
                        regime_align = 0.8  # ETFs align well with risk-on
                    elif current_regime == Regime.RISK_OFF:
                        regime_align = -0.5  # ETFs don't align well with risk-off
                elif opp["instrument"] == Instrument.OPTIONS.value:
                    if current_regime == Regime.RISK_ON:
                        regime_align = 0.5  # Options align somewhat with risk-on
                    elif current_regime == Regime.RISK_OFF:
                        regime_align = -0.8  # Options don't align well with risk-off
                elif opp["instrument"] == Instrument.CRYPTO.value:
                    if current_regime == Regime.RISK_ON:
                        regime_align = 0.9  # Crypto aligns very well with risk-on
                    elif current_regime == Regime.RISK_OFF:
                        regime_align = -0.9  # Crypto doesn't align well with risk-off
            else:
                regime_align = opp.get("regime_align", 0.0)
            
            # Create processed opportunity
            processed_opp: Opportunity = {
                "id": opp_id,
                "instrument": opp["instrument"],
                "symbol": opp["symbol"],
                "ts": ts,
                "alpha": opp.get("alpha", 0.0),
                "regime_align": regime_align,
                "sentiment_boost": opp.get("sentiment_boost", 0.0),
                "est_cost_bps": opp.get("est_cost_bps", 0.0),
                "risk_penalty": opp.get("risk_penalty", 0.0),
                "meta": opp.get("meta", {}),
                "ctx": {
                    "px": opp.get("price", 0.0),
                    "vol": opp.get("volume", 0.0),
                    "size_budget_usd": opp.get("size_budget_usd", 1000.0)
                }
            }
            
            result.append(processed_opp)
        
        return result
    
    def _execute_callbacks(self, opp: RoutedOpportunity) -> None:
        """
        Execute all registered callbacks for a routed opportunity.
        
        Args:
            opp: The routed opportunity
        """
        for callback in self.order_callbacks:
            try:
                callback(opp)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")
