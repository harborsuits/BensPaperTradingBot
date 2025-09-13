#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeFi Strategy Mixin

This module provides DeFi-specific capabilities for cryptocurrency strategies,
including on-chain analytics, protocol-specific functionality, and yield opportunities.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class DeFiStrategyMixin:
    """
    Mixin class that adds DeFi capabilities to crypto strategies.
    
    This mixin provides:
    1. On-chain data analysis
    2. DEX interaction (Uniswap, SushiSwap, etc.)
    3. Yield farming opportunities
    4. Liquidity provisioning
    5. MEV protection
    6. Gas optimization
    """
    
    def __init__(self, *args, **kwargs):
        # Call the parent's __init__ if it exists
        super().__init__(*args, **kwargs)
        
        # DeFi-specific state
        self.defi_protocols = {}  # Protocol -> data
        self.yield_opportunities = []
        self.liquidity_pools = []
        self.on_chain_metrics = {}
        self.mempool_data = {}
        
        # Gas price tracking
        self.current_gas_price = 0
        self.gas_price_history = []
        
        # Protocol-specific settings
        self.protocol_settings = {
            'uniswap': {
                'version': 3,
                'max_slippage': 0.005,  # 0.5%
                'use_flashbots': True,  # MEV protection
            },
            'aave': {
                'version': 3,
                'safety_module_stake': False,
                'use_stable_rate': False,
            },
            'compound': {
                'version': 3,
                'collateral_factor_buffer': 0.1,  # 10% buffer for liquidation
            }
        }
    
    def update_on_chain_data(self, on_chain_data: Dict[str, Any]) -> None:
        """
        Update on-chain data.
        
        Args:
            on_chain_data: Dictionary with on-chain metrics
        """
        self.on_chain_metrics = on_chain_data
        self.current_gas_price = on_chain_data.get('gas_price', 0)
        
        # Track gas price history
        self.gas_price_history.append({
            'timestamp': datetime.now(timezone.utc),
            'price': self.current_gas_price
        })
        
        logger.debug(f"Updated on-chain data. Gas price: {self.current_gas_price} gwei")
    
    def update_defi_protocol_data(self, protocol: str, data: Dict[str, Any]) -> None:
        """
        Update data for a specific DeFi protocol.
        
        Args:
            protocol: Protocol name (e.g., 'uniswap', 'aave')
            data: Protocol-specific data
        """
        self.defi_protocols[protocol] = data
        logger.debug(f"Updated {protocol} protocol data")
    
    def update_yield_opportunities(self, opportunities: List[Dict[str, Any]]) -> None:
        """
        Update yield farming opportunities.
        
        Args:
            opportunities: List of yield opportunities
        """
        self.yield_opportunities = opportunities
        logger.debug(f"Updated yield opportunities: {len(opportunities)} found")
    
    def update_liquidity_pools(self, pools: List[Dict[str, Any]]) -> None:
        """
        Update liquidity pool data.
        
        Args:
            pools: List of liquidity pools
        """
        self.liquidity_pools = pools
        logger.debug(f"Updated liquidity pools: {len(pools)} pools")
    
    def update_mempool_data(self, mempool_data: Dict[str, Any]) -> None:
        """
        Update mempool data for MEV analysis.
        
        Args:
            mempool_data: Dictionary with mempool transactions
        """
        self.mempool_data = mempool_data
        logger.debug(f"Updated mempool data: {len(mempool_data.get('transactions', []))} transactions")
    
    def analyze_on_chain_metrics(self) -> Dict[str, Any]:
        """
        Analyze on-chain metrics for trading signals.
        
        Returns:
            Dictionary of analysis results
        """
        if not self.on_chain_metrics:
            return {}
        
        results = {}
        
        # Analyze network activity
        tx_count = self.on_chain_metrics.get('transaction_count', 0)
        avg_tx_count = self.on_chain_metrics.get('avg_transaction_count', 0)
        
        if avg_tx_count > 0:
            network_activity = tx_count / avg_tx_count
            results['network_activity'] = network_activity
            
            # Signal increased volatility if network activity spikes
            if network_activity > 1.5:
                results['volatility_signal'] = 'high'
            elif network_activity < 0.7:
                results['volatility_signal'] = 'low'
            else:
                results['volatility_signal'] = 'normal'
        
        # Analyze smart contract interactions
        contract_calls = self.on_chain_metrics.get('contract_calls', {})
        
        # Check DEX volume
        dex_volume = sum(
            vol for protocol, vol in contract_calls.items() 
            if protocol in ['uniswap', 'sushiswap', 'balancer']
        )
        results['dex_volume'] = dex_volume
        
        # Check lending protocol activity
        lending_activity = sum(
            vol for protocol, vol in contract_calls.items() 
            if protocol in ['aave', 'compound', 'maker']
        )
        results['lending_activity'] = lending_activity
        
        # Detect whale movements
        whale_transfers = self.on_chain_metrics.get('whale_transfers', [])
        if whale_transfers:
            # Look for significant movements
            significant_movements = [
                t for t in whale_transfers 
                if t.get('value_usd', 0) > 1000000  # Over $1M transfers
            ]
            results['whale_activity'] = len(significant_movements)
            
            # Categorize whale activity
            if results['whale_activity'] > 3:
                results['whale_signal'] = 'high'
            elif results['whale_activity'] > 0:
                results['whale_signal'] = 'medium'
            else:
                results['whale_signal'] = 'low'
        
        # Gas price analysis
        if self.gas_price_history:
            current_gas = self.current_gas_price
            avg_gas = sum(g['price'] for g in self.gas_price_history[-10:]) / min(10, len(self.gas_price_history))
            results['gas_price_ratio'] = current_gas / avg_gas if avg_gas > 0 else 1.0
            
            # Signal for urgent transactions based on gas
            if results['gas_price_ratio'] > 2.0:
                results['gas_signal'] = 'high_congestion'
            elif results['gas_price_ratio'] > 1.3:
                results['gas_signal'] = 'moderate_congestion'
            else:
                results['gas_signal'] = 'normal'
        
        return results
    
    def find_best_yield_opportunity(self, 
                                   asset: str, 
                                   min_apy: float = 0.0, 
                                   max_risk: str = 'medium') -> Optional[Dict[str, Any]]:
        """
        Find the best yield opportunity for a specific asset.
        
        Args:
            asset: Asset symbol
            min_apy: Minimum APY required
            max_risk: Maximum risk level ('low', 'medium', 'high')
            
        Returns:
            Best opportunity or None
        """
        risk_levels = {'low': 1, 'medium': 2, 'high': 3}
        max_risk_level = risk_levels.get(max_risk, 2)
        
        # Filter opportunities by asset and minimum APY
        filtered = [
            o for o in self.yield_opportunities
            if o.get('asset') == asset and
               o.get('apy', 0) >= min_apy and
               risk_levels.get(o.get('risk', 'medium'), 2) <= max_risk_level
        ]
        
        # Sort by APY (descending)
        filtered.sort(key=lambda x: x.get('apy', 0), reverse=True)
        
        return filtered[0] if filtered else None
    
    def analyze_liquidity_pools(self, 
                              asset_pair: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze liquidity pools for trading opportunities.
        
        Args:
            asset_pair: Optional asset pair to filter (e.g., 'ETH/USDC')
            
        Returns:
            Analysis results
        """
        if not self.liquidity_pools:
            return {}
        
        results = {
            'pools_analyzed': 0,
            'price_impact_data': {},
            'impermanent_loss_estimates': {},
            'liquidity_distribution': {},
            'arbitrage_opportunities': []
        }
        
        for pool in self.liquidity_pools:
            pool_pair = pool.get('pair')
            
            # Skip if we're filtering for a specific pair and this isn't it
            if asset_pair and pool_pair != asset_pair:
                continue
                
            # Track the pool in our analysis
            results['pools_analyzed'] += 1
            
            # Analyze price impact
            results['price_impact_data'][pool_pair] = {
                'trade_size_small': pool.get('price_impact_0.1', 0),
                'trade_size_medium': pool.get('price_impact_1.0', 0),
                'trade_size_large': pool.get('price_impact_10.0', 0)
            }
            
            # Analyze impermanent loss risk
            if 'price_24h_change' in pool and 'liquidity' in pool:
                # Calculate estimated IL based on price change
                price_change = pool['price_24h_change']
                # Simplified IL formula
                il_estimate = 2 * (price_change**0.5) / (1 + price_change) - 1
                results['impermanent_loss_estimates'][pool_pair] = abs(il_estimate)
            
            # Check for arbitrage
            if 'price' in pool and 'external_price' in pool:
                price_diff = (pool['price'] - pool['external_price']) / pool['external_price']
                
                # If price difference is over 0.5%, consider it an arbitrage opportunity
                if abs(price_diff) > 0.005:
                    results['arbitrage_opportunities'].append({
                        'pair': pool_pair,
                        'pool': pool.get('protocol', 'unknown'),
                        'price_diff_pct': price_diff * 100,
                        'estimated_profit': price_diff * pool.get('liquidity', 0) * 0.1,  # 10% depth utilization
                        'direction': 'buy' if price_diff < 0 else 'sell'
                    })
        
        return results
    
    def analyze_lending_protocols(self) -> Dict[str, Any]:
        """
        Analyze lending protocols for opportunities and risks.
        
        Returns:
            Analysis results
        """
        results = {
            'lending_rates': {},
            'borrowing_rates': {},
            'liquidation_risks': {},
            'best_supply_opportunities': [],
            'best_borrow_opportunities': []
        }
        
        for protocol, data in self.defi_protocols.items():
            if protocol not in ['aave', 'compound', 'maker']:
                continue
                
            markets = data.get('markets', [])
            for market in markets:
                asset = market.get('asset')
                
                # Track lending rates
                supply_rate = market.get('supply_rate', 0)
                if supply_rate > 0:
                    if asset not in results['lending_rates']:
                        results['lending_rates'][asset] = []
                    results['lending_rates'][asset].append({
                        'protocol': protocol,
                        'rate': supply_rate,
                        'liquidity': market.get('liquidity', 0)
                    })
                
                # Track borrowing rates
                borrow_rate = market.get('borrow_rate', 0)
                if borrow_rate > 0:
                    if asset not in results['borrowing_rates']:
                        results['borrowing_rates'][asset] = []
                    results['borrowing_rates'][asset].append({
                        'protocol': protocol,
                        'rate': borrow_rate,
                        'available': market.get('available_to_borrow', 0)
                    })
                
                # Assess liquidation risks
                if 'utilization_rate' in market and 'liquidation_threshold' in market:
                    util_rate = market['utilization_rate']
                    liq_threshold = market['liquidation_threshold']
                    
                    # Higher utilization rates increase liquidation risk
                    risk_level = 'low'
                    if util_rate > 0.9:
                        risk_level = 'high'
                    elif util_rate > 0.75:
                        risk_level = 'medium'
                        
                    results['liquidation_risks'][f"{asset}_{protocol}"] = {
                        'risk_level': risk_level,
                        'utilization': util_rate,
                        'threshold': liq_threshold
                    }
        
        # Find best opportunities
        for asset, rates in results['lending_rates'].items():
            if rates:
                # Sort by rate (descending)
                rates.sort(key=lambda x: x['rate'], reverse=True)
                results['best_supply_opportunities'].append({
                    'asset': asset,
                    'protocol': rates[0]['protocol'],
                    'rate': rates[0]['rate']
                })
        
        for asset, rates in results['borrowing_rates'].items():
            if rates:
                # Sort by rate (ascending - we want cheapest borrowing)
                rates.sort(key=lambda x: x['rate'])
                results['best_borrow_opportunities'].append({
                    'asset': asset,
                    'protocol': rates[0]['protocol'],
                    'rate': rates[0]['rate']
                })
        
        return results
    
    def analyze_mempool_for_mev(self) -> Dict[str, Any]:
        """
        Analyze mempool data for MEV opportunities and risks.
        
        Returns:
            Analysis results
        """
        if not self.mempool_data:
            return {}
        
        results = {
            'sandwich_attack_risk': 'low',
            'frontrunning_risk': 'low',
            'arbitrage_opportunities': [],
            'high_value_txs': []
        }
        
        transactions = self.mempool_data.get('transactions', [])
        
        # Check for pending swap transactions with high value
        swap_txs = [tx for tx in transactions if tx.get('type') == 'swap']
        high_value_swaps = [tx for tx in swap_txs if tx.get('value_usd', 0) > 50000]  # Over $50k
        
        if high_value_swaps:
            results['high_value_txs'] = [
                {
                    'hash': tx.get('hash', 'unknown'),
                    'from': tx.get('from', 'unknown'),
                    'value_usd': tx.get('value_usd', 0),
                    'gas_price': tx.get('gas_price', 0),
                    'type': tx.get('type', 'unknown'),
                    'pair': tx.get('pair', 'unknown')
                }
                for tx in high_value_swaps[:5]  # Top 5 by value
            ]
        
        # Check for sandwich attack risk
        if len(swap_txs) > 5:
            # Check for transactions with similar pairs but different gas prices
            pairs = {}
            for tx in swap_txs:
                pair = tx.get('pair')
                if pair:
                    if pair not in pairs:
                        pairs[pair] = []
                    pairs[pair].append(tx)
            
            # Pairs with multiple pending transactions are at risk
            pairs_at_risk = {}
            for pair, pair_txs in pairs.items():
                if len(pair_txs) > 2:
                    # Check for gas price disparity
                    gas_prices = [tx.get('gas_price', 0) for tx in pair_txs]
                    if max(gas_prices) > min(gas_prices) * 1.5:  # 50% gas price disparity
                        pairs_at_risk[pair] = len(pair_txs)
            
            if pairs_at_risk:
                results['sandwich_attack_risk'] = 'high' if len(pairs_at_risk) > 2 else 'medium'
        
        # Check for frontrunning risk
        if transactions:
            gas_prices = [tx.get('gas_price', 0) for tx in transactions]
            if gas_prices:
                avg_gas = sum(gas_prices) / len(gas_prices)
                high_gas_txs = [tx for tx in transactions if tx.get('gas_price', 0) > avg_gas * 1.5]
                
                if len(high_gas_txs) > 3:  # Multiple high gas price transactions
                    results['frontrunning_risk'] = 'high'
                elif high_gas_txs:
                    results['frontrunning_risk'] = 'medium'
        
        return results
    
    def estimate_gas_costs(self, 
                          transaction_type: str, 
                          fast: bool = False) -> Dict[str, float]:
        """
        Estimate gas costs for different transaction types.
        
        Args:
            transaction_type: Type of transaction (swap, add_liquidity, etc.)
            fast: Whether to use fast gas price
            
        Returns:
            Gas cost estimates
        """
        base_gas_price = self.current_gas_price
        if fast:
            base_gas_price *= 1.5
        
        # Gas units used by different transaction types
        gas_units = {
            'swap': 150000,
            'add_liquidity': 200000,
            'remove_liquidity': 160000,
            'stake': 120000,
            'unstake': 100000,
            'borrow': 220000,
            'repay': 120000,
            'claim_rewards': 90000,
            'approve': 45000
        }
        
        gas_units_used = gas_units.get(transaction_type, 100000)
        
        # Assuming 1 ETH = $2800
        eth_price_usd = 2800
        gas_cost_eth = (base_gas_price * gas_units_used) / 1e9  # Convert wei to ETH
        gas_cost_usd = gas_cost_eth * eth_price_usd
        
        return {
            'gas_price_gwei': base_gas_price,
            'gas_units': gas_units_used,
            'gas_cost_eth': gas_cost_eth,
            'gas_cost_usd': gas_cost_usd
        }
    
    def check_optimal_execution_time(self) -> Dict[str, Any]:
        """
        Check if current gas price indicates optimal time to execute transactions.
        
        Returns:
            Dictionary with execution recommendations
        """
        if not self.gas_price_history:
            return {'should_execute': True, 'reason': 'No gas history available'}
        
        # Get last 24h of gas price data if available
        recent_history = self.gas_price_history[-24:] if len(self.gas_price_history) >= 24 else self.gas_price_history
        
        avg_gas = sum(entry['price'] for entry in recent_history) / len(recent_history)
        lowest_gas = min(entry['price'] for entry in recent_history)
        highest_gas = max(entry['price'] for entry in recent_history)
        
        current_gas = self.current_gas_price
        
        # Calculate where current gas falls in the range
        gas_percentile = (current_gas - lowest_gas) / (highest_gas - lowest_gas) if (highest_gas - lowest_gas) > 0 else 0.5
        
        if current_gas <= avg_gas * 0.8:
            return {
                'should_execute': True,
                'reason': 'Gas price is significantly below average',
                'gas_price': current_gas,
                'avg_gas_price': avg_gas,
                'gas_percentile': gas_percentile
            }
        elif current_gas <= avg_gas:
            return {
                'should_execute': True,
                'reason': 'Gas price is below average',
                'gas_price': current_gas,
                'avg_gas_price': avg_gas,
                'gas_percentile': gas_percentile
            }
        elif current_gas <= avg_gas * 1.2:
            return {
                'should_execute': True,
                'reason': 'Gas price is slightly above average but acceptable',
                'gas_price': current_gas,
                'avg_gas_price': avg_gas,
                'gas_percentile': gas_percentile
            }
        else:
            return {
                'should_execute': False,
                'reason': 'Gas price is too high',
                'gas_price': current_gas,
                'avg_gas_price': avg_gas,
                'gas_percentile': gas_percentile,
                'recommendation': 'Wait for lower gas prices'
            }
