#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeFi Yield Farming Strategy

A strategy that optimizes yield farming across DeFi protocols, with smart
rebalancing, impermanent loss protection, and gas-efficient operations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from trading_bot.strategies_new.crypto.base import CryptoBaseStrategy, CryptoSession
from trading_bot.strategies_new.crypto.mixins.crypto_account_aware_mixin import CryptoAccountAwareMixin
from trading_bot.strategies_new.crypto.mixins.defi_strategy_mixin import DeFiStrategyMixin
from trading_bot.core.events import Event, EventType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.factory.registry import register_strategy

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="YieldFarmingStrategy",
    market_type="crypto",
    description="Advanced DeFi yield farming strategy with optimal protocol allocation and IL protection",
    timeframes=["H1", "H4", "D1"],
    parameters={
        # Strategy parameters
        "rebalance_frequency_hours": {"type": "int", "default": 24, "min": 1, "max": 168},
        "min_apy_threshold": {"type": "float", "default": 5.0, "min": 1.0, "max": 100.0},
        "max_gas_per_tx_usd": {"type": "float", "default": 50.0, "min": 1.0, "max": 500.0},
        "risk_profile": {"type": "string", "default": "medium", "enum": ["low", "medium", "high"]},
        
        # Position allocation
        "max_allocation_per_protocol_pct": {"type": "float", "default": 30.0, "min": 5.0, "max": 50.0},
        "max_allocation_per_asset_pct": {"type": "float", "default": 25.0, "min": 5.0, "max": 50.0},
        "stable_asset_minimum_pct": {"type": "float", "default": 20.0, "min": 0.0, "max": 80.0},
        
        # Risk management
        "impermanent_loss_threshold_pct": {"type": "float", "default": 2.0, "min": 0.5, "max": 10.0},
        "price_volatility_exit_threshold": {"type": "float", "default": 10.0, "min": 5.0, "max": 30.0},
        "withdraw_on_adverse_news": {"type": "bool", "default": True},
        
        # Protocol preferences
        "preferred_protocols": {"type": "array", "default": ["aave", "compound", "uniswap", "curve"]},
        "preferred_networks": {"type": "array", "default": ["ethereum", "polygon", "arbitrum", "optimism"]},
    }
)
class YieldFarmingStrategy(CryptoBaseStrategy, CryptoAccountAwareMixin, DeFiStrategyMixin):
    """
    An advanced DeFi yield farming strategy that:
    1. Allocates funds across multiple protocols to optimize APY
    2. Considers gas costs and transaction timing for efficiency
    3. Monitors impermanent loss risk and rebalances accordingly
    4. Implements protocol-specific optimizations
    5. Uses on-chain metrics to make dynamic allocation decisions
    """
    
    def __init__(self, session: CryptoSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the yield farming strategy.
        
        Args:
            session: Crypto trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters
        """
        CryptoBaseStrategy.__init__(self, session, data_pipeline, parameters)
        CryptoAccountAwareMixin.__init__(self)
        DeFiStrategyMixin.__init__(self)
        
        # Initialize strategy state
        self.active_positions = []  # Track active yield positions
        self.last_rebalance_time = None
        self.last_opportunity_scan = None
        self.pending_transactions = []
        self.protocol_performance = {}  # Historical APY by protocol
        
        # Set up protocol-specific handlers
        self.protocol_handlers = {
            "aave": self._handle_aave_operations,
            "compound": self._handle_compound_operations,
            "uniswap": self._handle_uniswap_operations,
            "curve": self._handle_curve_operations,
            "convex": self._handle_convex_operations,
        }
        
        # Opportunity tracking
        self.current_opportunities = []
        self.historical_opportunities = []
        
        logger.info(f"Initialized {self.name} with risk profile: {self.parameters['risk_profile']}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators for yield farming.
        
        For yield farming, we're more interested in on-chain metrics, protocol TVL,
        and APY trends rather than traditional technical indicators.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Calculate basic market indicators that might affect yield
        if len(data) > 20:
            # Calculate volatility (20-day)
            indicators['price_volatility'] = data['close'].pct_change().rolling(20).std() * 100
            
            # Calculate trend strength
            indicators['trend_strength'] = (
                (data['close'].iloc[-1] - data['close'].iloc[-20]) / 
                data['close'].iloc[-20] * 100
            )
            
            # Calculate market momentum
            indicators['momentum'] = (
                data['close'].iloc[-1] - data['close'].iloc[-10]
            ) / data['close'].iloc[-10] * 100
        
        # Add on-chain indicators if available
        if hasattr(self, 'on_chain_metrics') and self.on_chain_metrics:
            on_chain_analysis = self.analyze_on_chain_metrics()
            indicators.update(on_chain_analysis)
        
        # Add protocol-specific indicators if available
        if hasattr(self, 'defi_protocols') and self.defi_protocols:
            # Analyze lending protocols
            lending_analysis = self.analyze_lending_protocols()
            
            # Extract key metrics
            if 'best_supply_opportunities' in lending_analysis:
                indicators['best_supply_opportunities'] = lending_analysis['best_supply_opportunities']
            
            # Get average lending rates across protocols
            if 'lending_rates' in lending_analysis:
                avg_rates = {}
                for asset, rates in lending_analysis['lending_rates'].items():
                    if rates:
                        avg_rates[asset] = sum(r['rate'] for r in rates) / len(rates)
                indicators['avg_lending_rates'] = avg_rates
        
        # Add liquidity pool analysis
        if hasattr(self, 'liquidity_pools') and self.liquidity_pools:
            lp_analysis = self.analyze_liquidity_pools()
            
            # Extract key LP metrics
            if 'impermanent_loss_estimates' in lp_analysis:
                indicators['impermanent_loss_estimates'] = lp_analysis['impermanent_loss_estimates']
            
            if 'arbitrage_opportunities' in lp_analysis:
                indicators['arbitrage_opportunities'] = lp_analysis['arbitrage_opportunities']
        
        # Add gas price indicators
        if hasattr(self, 'current_gas_price'):
            indicators['current_gas_price'] = self.current_gas_price
            
            # Check optimal execution time
            execution_recommendation = self.check_optimal_execution_time()
            indicators['should_execute'] = execution_recommendation.get('should_execute', True)
            indicators['gas_execution_reason'] = execution_recommendation.get('reason', '')
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on yield opportunities.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            'actions': [],
            'allocations': {},
            'protocols_to_enter': [],
            'protocols_to_exit': [],
            'should_rebalance': False
        }
        
        # Check if it's time to rebalance
        rebalance_hours = self.parameters.get('rebalance_frequency_hours', 24)
        
        if (self.last_rebalance_time is None or 
            datetime.now(timezone.utc) - self.last_rebalance_time > timedelta(hours=rebalance_hours)):
            signals['should_rebalance'] = True
        
        # Check if gas prices are acceptable for operations
        should_execute = indicators.get('should_execute', True)
        current_gas_price = indicators.get('current_gas_price', 0)
        
        # Skip operations if gas is too high unless we're in a risk situation
        high_risk_situation = False
        il_estimates = indicators.get('impermanent_loss_estimates', {})
        
        # Check if any position has high impermanent loss risk
        for pair, il in il_estimates.items():
            if il > self.parameters.get('impermanent_loss_threshold_pct', 2.0) / 100:
                high_risk_situation = True
                signals['actions'].append(f"Exit {pair} LP position due to high IL risk ({il:.2%})")
                
                # Find the protocol for this pair
                for position in self.active_positions:
                    if position.get('pair') == pair:
                        protocol = position.get('protocol')
                        if protocol and protocol not in signals['protocols_to_exit']:
                            signals['protocols_to_exit'].append(protocol)
        
        # If gas is too high and we're not in a risk situation, suggest waiting
        if not should_execute and not high_risk_situation:
            signals['actions'].append(f"Defer operations due to high gas ({current_gas_price} gwei)")
            signals['should_rebalance'] = False
        
        # Find the best yield opportunities
        best_opportunities = []
        
        # Check lending opportunities
        if 'best_supply_opportunities' in indicators:
            for opportunity in indicators['best_supply_opportunities']:
                apy = opportunity.get('rate', 0) * 100  # Convert to percentage
                if apy >= self.parameters.get('min_apy_threshold', 5.0):
                    best_opportunities.append({
                        'type': 'lending',
                        'asset': opportunity.get('asset'),
                        'protocol': opportunity.get('protocol'),
                        'apy': apy,
                        'risk': 'low'  # Lending is generally lower risk
                    })
        
        # Check LP opportunities if we have liquidity pools
        if hasattr(self, 'liquidity_pools') and self.liquidity_pools:
            for pool in self.liquidity_pools:
                pair = pool.get('pair')
                apy = pool.get('apy', 0)
                
                # Check if this pool's APY meets our threshold
                if apy >= self.parameters.get('min_apy_threshold', 5.0):
                    # Check impermanent loss risk
                    il_risk = il_estimates.get(pair, 0)
                    risk_level = 'medium'
                    
                    if il_risk > 0.05:  # More than 5% IL risk
                        risk_level = 'high'
                    elif il_risk < 0.01:  # Less than 1% IL risk
                        risk_level = 'low'
                    
                    best_opportunities.append({
                        'type': 'liquidity_pool',
                        'pair': pair,
                        'protocol': pool.get('protocol'),
                        'apy': apy,
                        'risk': risk_level,
                        'il_risk': il_risk
                    })
        
        # Filter opportunities based on risk profile
        risk_profile = self.parameters.get('risk_profile', 'medium')
        filtered_opportunities = []
        
        for opportunity in best_opportunities:
            opportunity_risk = opportunity.get('risk', 'medium')
            
            # Apply risk filtering
            if risk_profile == 'low' and opportunity_risk != 'low':
                continue
            elif risk_profile == 'medium' and opportunity_risk == 'high':
                continue
            # For high risk profile, accept all opportunities
            
            filtered_opportunities.append(opportunity)
        
        # Sort by APY (descending)
        filtered_opportunities.sort(key=lambda x: x.get('apy', 0), reverse=True)
        
        # Update signals with top opportunities
        signals['top_opportunities'] = filtered_opportunities[:5]  # Top 5 opportunities
        
        # Determine which protocols to enter
        for opportunity in filtered_opportunities[:3]:  # Consider top 3 for allocation
            protocol = opportunity.get('protocol')
            if protocol and protocol not in signals['protocols_to_enter']:
                signals['protocols_to_enter'].append(protocol)
        
        # Calculate optimal allocations
        if filtered_opportunities and signals['should_rebalance']:
            total_score = sum(opp.get('apy', 0) for opp in filtered_opportunities[:5])
            
            if total_score > 0:
                for i, opp in enumerate(filtered_opportunities[:5]):
                    allocation = (opp.get('apy', 0) / total_score) * 100
                    max_allocation = self.parameters.get('max_allocation_per_protocol_pct', 30.0)
                    
                    # Cap allocation to maximum
                    allocation = min(allocation, max_allocation)
                    
                    # Adjust for risk
                    if opp.get('risk') == 'high':
                        allocation *= 0.7  # Reduce allocation for high-risk opportunities
                    elif opp.get('risk') == 'low':
                        allocation *= 1.2  # Increase allocation for low-risk opportunities
                    
                    key = f"{opp.get('protocol')}_{opp.get('asset', opp.get('pair', 'unknown'))}"
                    signals['allocations'][key] = allocation
        
        # Save current opportunities for tracking
        self.current_opportunities = filtered_opportunities
        
        return signals
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size for yield farming.
        
        For yield farming, we calculate based on optimal allocation percentages
        and risk parameters rather than traditional position sizing.
        
        Args:
            direction: Not used for yield farming
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size in crypto units (not used directly for yield farming)
        """
        # For yield farming, this is handled differently
        # We'll return a placeholder value
        return 0.0
    
    def _check_for_trade_opportunities(self) -> None:
        """
        Check for yield farming opportunities.
        
        This overrides the base method to implement yield-specific logic.
        """
        if not self.is_active or not self.market_data.empty:
            return
        
        # Calculate indicators and generate signals
        indicators = self.calculate_indicators(self.market_data)
        signals = self.generate_signals(self.market_data, indicators)
        
        # Check if we should rebalance
        if signals.get('should_rebalance', False):
            logger.info("Rebalancing yield farming positions")
            
            # Exit protocols as needed
            for protocol in signals.get('protocols_to_exit', []):
                self._exit_protocol_positions(protocol)
            
            # Enter new protocols based on allocations
            allocations = signals.get('allocations', {})
            self._apply_allocations(allocations)
            
            # Update rebalance time
            self.last_rebalance_time = datetime.now(timezone.utc)
        
        # Process any pending transactions
        self._process_pending_transactions()
    
    def _exit_protocol_positions(self, protocol: str) -> None:
        """
        Exit positions for a specific protocol.
        
        Args:
            protocol: Protocol name
        """
        positions_to_remove = []
        
        for position in self.active_positions:
            if position.get('protocol') == protocol:
                # Invoke the appropriate protocol handler
                handler = self.protocol_handlers.get(protocol.lower())
                if handler:
                    tx_data = handler('exit', position)
                    self.pending_transactions.append(tx_data)
                    positions_to_remove.append(position)
        
        # Remove exited positions
        for position in positions_to_remove:
            self.active_positions.remove(position)
        
        logger.info(f"Exited {len(positions_to_remove)} positions from {protocol}")
    
    def _apply_allocations(self, allocations: Dict[str, float]) -> None:
        """
        Apply allocation percentages to yield farming positions.
        
        Args:
            allocations: Dictionary mapping protocol_asset to allocation percentage
        """
        portfolio_value = self._calculate_portfolio_value()
        
        for key, allocation_pct in allocations.items():
            parts = key.split('_')
            if len(parts) >= 2:
                protocol = parts[0]
                asset = '_'.join(parts[1:])  # Handle asset names that might contain underscores
                
                # Calculate allocation amount
                allocation_amount = portfolio_value * (allocation_pct / 100)
                
                # Check if we can afford gas for this operation
                gas_estimate = self.estimate_gas_costs('add_liquidity', fast=True)
                if gas_estimate.get('gas_cost_usd', 0) > self.parameters.get('max_gas_per_tx_usd', 50.0):
                    logger.warning(f"Skipping {protocol}_{asset} allocation due to high gas costs")
                    continue
                
                # Invoke the appropriate protocol handler
                handler = self.protocol_handlers.get(protocol.lower())
                if handler:
                    position_data = {
                        'protocol': protocol,
                        'asset': asset,
                        'allocation_pct': allocation_pct,
                        'allocation_amount': allocation_amount
                    }
                    
                    tx_data = handler('enter', position_data)
                    self.pending_transactions.append(tx_data)
                    
                    # Add to active positions
                    self.active_positions.append(position_data)
                    
                    logger.info(f"Allocated {allocation_amount:.2f} USD ({allocation_pct:.1f}%) to {protocol}_{asset}")
    
    def _process_pending_transactions(self) -> None:
        """Process pending yield farming transactions."""
        # In a real implementation, this would poll for transaction status
        # and update positions based on confirmed transactions
        completed_txs = []
        
        for tx in self.pending_transactions:
            # Simulate transaction completion
            # In reality, this would check the blockchain for confirmation
            if tx.get('status') == 'pending':
                tx['status'] = 'completed'
                logger.info(f"Completed transaction: {tx.get('type')} on {tx.get('protocol')}")
                completed_txs.append(tx)
        
        # Remove completed transactions
        for tx in completed_txs:
            self.pending_transactions.remove(tx)
    
    def _handle_aave_operations(self, operation: str, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle Aave-specific operations.
        
        Args:
            operation: 'enter' or 'exit'
            position_data: Position details
            
        Returns:
            Transaction data
        """
        if operation == 'enter':
            # For lending on Aave
            return {
                'type': 'supply',
                'protocol': 'aave',
                'asset': position_data.get('asset'),
                'amount': position_data.get('allocation_amount'),
                'status': 'pending',
                'apy_expected': self._get_aave_supply_rate(position_data.get('asset')),
                'timestamp': datetime.now(timezone.utc)
            }
        else:  # exit
            return {
                'type': 'withdraw',
                'protocol': 'aave',
                'asset': position_data.get('asset'),
                'amount': position_data.get('allocation_amount'),
                'status': 'pending',
                'timestamp': datetime.now(timezone.utc)
            }
    
    def _handle_compound_operations(self, operation: str, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Compound-specific operations."""
        if operation == 'enter':
            return {
                'type': 'supply',
                'protocol': 'compound',
                'asset': position_data.get('asset'),
                'amount': position_data.get('allocation_amount'),
                'status': 'pending',
                'apy_expected': self._get_compound_supply_rate(position_data.get('asset')),
                'timestamp': datetime.now(timezone.utc)
            }
        else:  # exit
            return {
                'type': 'withdraw',
                'protocol': 'compound',
                'asset': position_data.get('asset'),
                'amount': position_data.get('allocation_amount'),
                'status': 'pending',
                'timestamp': datetime.now(timezone.utc)
            }
    
    def _handle_uniswap_operations(self, operation: str, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Uniswap LP operations."""
        asset = position_data.get('asset')
        
        # Extract token pair from asset
        token_a, token_b = asset.split('-') if '-' in asset else (asset, 'ETH')
        
        if operation == 'enter':
            return {
                'type': 'add_liquidity',
                'protocol': 'uniswap',
                'pair': f"{token_a}-{token_b}",
                'amount': position_data.get('allocation_amount'),
                'token_a': token_a,
                'token_b': token_b,
                'status': 'pending',
                'apy_expected': self._get_uniswap_lp_apy(token_a, token_b),
                'timestamp': datetime.now(timezone.utc)
            }
        else:  # exit
            return {
                'type': 'remove_liquidity',
                'protocol': 'uniswap',
                'pair': position_data.get('pair', f"{token_a}-{token_b}"),
                'amount': position_data.get('allocation_amount'),
                'status': 'pending',
                'timestamp': datetime.now(timezone.utc)
            }
    
    def _handle_curve_operations(self, operation: str, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Curve operations."""
        pool = position_data.get('asset')
        
        if operation == 'enter':
            return {
                'type': 'add_liquidity',
                'protocol': 'curve',
                'pool': pool,
                'amount': position_data.get('allocation_amount'),
                'status': 'pending',
                'apy_expected': self._get_curve_apy(pool),
                'timestamp': datetime.now(timezone.utc)
            }
        else:  # exit
            return {
                'type': 'remove_liquidity',
                'protocol': 'curve',
                'pool': pool,
                'amount': position_data.get('allocation_amount'),
                'status': 'pending',
                'timestamp': datetime.now(timezone.utc)
            }
    
    def _handle_convex_operations(self, operation: str, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Convex operations."""
        pool = position_data.get('asset')
        
        if operation == 'enter':
            return {
                'type': 'stake',
                'protocol': 'convex',
                'pool': pool,
                'amount': position_data.get('allocation_amount'),
                'status': 'pending',
                'apy_expected': self._get_convex_apy(pool),
                'timestamp': datetime.now(timezone.utc)
            }
        else:  # exit
            return {
                'type': 'unstake',
                'protocol': 'convex',
                'pool': pool,
                'amount': position_data.get('allocation_amount'),
                'status': 'pending',
                'timestamp': datetime.now(timezone.utc)
            }
    
    def _get_aave_supply_rate(self, asset: str) -> float:
        """Get the current supply rate for an asset on Aave."""
        if 'aave' in self.defi_protocols:
            aave_data = self.defi_protocols['aave']
            markets = aave_data.get('markets', [])
            
            for market in markets:
                if market.get('asset') == asset:
                    return market.get('supply_rate', 0.0)
        
        return 0.0
    
    def _get_compound_supply_rate(self, asset: str) -> float:
        """Get the current supply rate for an asset on Compound."""
        if 'compound' in self.defi_protocols:
            compound_data = self.defi_protocols['compound']
            markets = compound_data.get('markets', [])
            
            for market in markets:
                if market.get('asset') == asset:
                    return market.get('supply_rate', 0.0)
        
        return 0.0
    
    def _get_uniswap_lp_apy(self, token_a: str, token_b: str) -> float:
        """Get the estimated APY for a Uniswap LP position."""
        pair = f"{token_a}-{token_b}"
        
        for pool in self.liquidity_pools:
            if pool.get('pair') == pair and pool.get('protocol') == 'uniswap':
                return pool.get('apy', 0.0)
        
        return 0.0
    
    def _get_curve_apy(self, pool: str) -> float:
        """Get the estimated APY for a Curve pool."""
        for lp in self.liquidity_pools:
            if lp.get('pool') == pool and lp.get('protocol') == 'curve':
                return lp.get('apy', 0.0)
        
        return 0.0
    
    def _get_convex_apy(self, pool: str) -> float:
        """Get the estimated APY for a Convex pool."""
        # Convex typically boosts Curve pools
        base_apy = self._get_curve_apy(pool)
        
        # Add Convex boost (typically 10-30%)
        convex_boost = 0.2  # 20% boost
        
        return base_apy * (1 + convex_boost)
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate compatibility with market regime.
        
        Yield farming can adapt to various market conditions, but works best
        in stable or low-volatility environments.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            "ranging": 0.95,        # Excellent in ranging markets
            "volatile": 0.60,       # Acceptable in volatile markets (impermanent loss risk)
            "trending": 0.75,       # Good in trending markets (may miss upside)
            "calm": 0.95,           # Excellent in calm markets
            "breakout": 0.40,       # Poor during breakouts (better to hold)
            "high_volume": 0.80,    # Good in high volume (trading fees for LPs)
            "low_volume": 0.70,     # Acceptable in low volume
            "high_liquidity": 0.90, # Very good in high liquidity markets
            "low_liquidity": 0.60,  # Acceptable in low liquidity (higher APYs but more risk)
        }
        
        return compatibility_map.get(market_regime, 0.75)
