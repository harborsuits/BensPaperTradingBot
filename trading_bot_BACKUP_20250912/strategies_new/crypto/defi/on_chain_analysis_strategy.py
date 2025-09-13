#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
On-Chain Analysis Strategy

A strategy that uses blockchain data and on-chain metrics to identify
trading opportunities in cryptocurrency markets.
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
    name="OnChainAnalysisStrategy",
    market_type="crypto",
    description="Trading strategy based on on-chain metrics and blockchain analytics",
    timeframes=["H1", "H4", "D1"],
    parameters={
        # On-chain metrics parameters
        "whale_transaction_threshold": {"type": "float", "default": 1000000.0, "min": 10000.0, "max": 10000000.0},
        "exchange_inflow_outflow_ratio_threshold": {"type": "float", "default": 1.5, "min": 1.1, "max": 3.0},
        "smart_money_follow_threshold": {"type": "float", "default": 0.7, "min": 0.5, "max": 0.9},
        
        # Trading parameters
        "signal_confirmation_count": {"type": "int", "default": 2, "min": 1, "max": 5},
        "stop_loss_pct": {"type": "float", "default": 5.0, "min": 1.0, "max": 15.0},
        "take_profit_pct": {"type": "float", "default": 10.0, "min": 2.0, "max": 30.0},
        "position_size_pct": {"type": "float", "default": 2.0, "min": 0.5, "max": 10.0},
        
        # Risk parameters
        "max_open_positions": {"type": "int", "default": 3, "min": 1, "max": 10},
        "max_concentration_pct": {"type": "float", "default": 15.0, "min": 5.0, "max": 30.0}
    }
)
class OnChainAnalysisStrategy(CryptoBaseStrategy, CryptoAccountAwareMixin, DeFiStrategyMixin):
    """
    A strategy that uses on-chain data to identify trading opportunities.
    
    This strategy:
    1. Monitors whale transactions and exchange flows
    2. Tracks smart contract interactions and protocol activity
    3. Analyzes network metrics like transaction counts and gas usage
    4. Follows "smart money" wallets and their trading behavior
    5. Correlates on-chain metrics with price action
    """
    
    def __init__(self, session: CryptoSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """Initialize the on-chain analysis strategy."""
        CryptoBaseStrategy.__init__(self, session, data_pipeline, parameters)
        CryptoAccountAwareMixin.__init__(self)
        DeFiStrategyMixin.__init__(self)
        
        # On-chain specific state
        self.whale_transactions = []
        self.exchange_flows = {}
        self.smart_money_wallets = {}
        self.network_metrics = {}
        self.contract_interactions = {}
        
        # Strategy state
        self.active_signals = {}
        self.confirmed_signals = {}
        self.signal_history = []
        
        # Smart wallets to track (could be loaded from config)
        self.tracked_wallets = [
            {"address": "0x1234567890abcdef1234567890abcdef12345678", "name": "Wallet A", "weight": 10},
            {"address": "0xabcdef1234567890abcdef1234567890abcdef12", "name": "Wallet B", "weight": 7},
            {"address": "0x7890abcdef1234567890abcdef1234567890abcd", "name": "Wallet C", "weight": 5}
        ]
        
        logger.info(f"Initialized {self.name}")
    
    def update_on_chain_data(self, on_chain_data: Dict[str, Any]) -> None:
        """
        Update on-chain data specific to this strategy.
        
        Args:
            on_chain_data: Dictionary with on-chain metrics
        """
        # Call parent method first
        super().update_on_chain_data(on_chain_data)
        
        # Update strategy-specific data
        self.whale_transactions = on_chain_data.get('whale_transactions', [])
        self.exchange_flows = on_chain_data.get('exchange_flows', {})
        self.network_metrics = on_chain_data.get('network_metrics', {})
        self.contract_interactions = on_chain_data.get('contract_interactions', {})
        
        # Update smart money wallet activity
        if 'wallet_activity' in on_chain_data:
            self._update_smart_money_wallets(on_chain_data['wallet_activity'])
        
        logger.debug(f"Updated on-chain data. Whale transactions: {len(self.whale_transactions)}")
    
    def _update_smart_money_wallets(self, wallet_activity: Dict[str, Any]) -> None:
        """
        Update smart money wallet tracking.
        
        Args:
            wallet_activity: Dictionary mapping wallet addresses to their activity
        """
        for wallet in self.tracked_wallets:
            address = wallet['address']
            if address in wallet_activity:
                activity = wallet_activity[address]
                
                if address not in self.smart_money_wallets:
                    self.smart_money_wallets[address] = {
                        'name': wallet.get('name', 'Unknown'),
                        'weight': wallet.get('weight', 5),
                        'history': []
                    }
                
                # Add new activity to history
                self.smart_money_wallets[address]['history'].append({
                    'timestamp': datetime.now(timezone.utc),
                    'activity': activity
                })
                
                # Keep only recent history (last 30 days)
                thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
                self.smart_money_wallets[address]['history'] = [
                    h for h in self.smart_money_wallets[address]['history']
                    if h['timestamp'] >= thirty_days_ago
                ]
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators based on on-chain data and market data.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Calculate basic market indicators
        if len(data) > 20:
            # Calculate price momentum
            indicators['price_momentum'] = (
                data['close'].iloc[-1] - data['close'].iloc[-10]
            ) / data['close'].iloc[-10] * 100
            
            # Calculate volume trend
            indicators['volume_trend'] = (
                data['volume'].iloc[-5:].mean() - data['volume'].iloc[-20:-5].mean()
            ) / data['volume'].iloc[-20:-5].mean() * 100
        
        # Analyze whale transactions
        if self.whale_transactions:
            # Count recent whale transactions
            recent_transactions = [
                t for t in self.whale_transactions
                if t.get('timestamp', datetime.now(timezone.utc)) > datetime.now(timezone.utc) - timedelta(days=1)
            ]
            
            indicators['recent_whale_count'] = len(recent_transactions)
            
            # Calculate net whale flow
            inflows = sum(t.get('value', 0) for t in recent_transactions if t.get('direction') == 'in')
            outflows = sum(t.get('value', 0) for t in recent_transactions if t.get('direction') == 'out')
            
            indicators['whale_net_flow'] = inflows - outflows
            indicators['whale_net_flow_ratio'] = inflows / outflows if outflows > 0 else float('inf')
        
        # Analyze exchange flows
        if self.exchange_flows:
            total_inflow = sum(flow.get('inflow', 0) for flow in self.exchange_flows.values())
            total_outflow = sum(flow.get('outflow', 0) for flow in self.exchange_flows.values())
            
            indicators['exchange_inflow'] = total_inflow
            indicators['exchange_outflow'] = total_outflow
            indicators['exchange_flow_ratio'] = total_inflow / total_outflow if total_outflow > 0 else float('inf')
            
            # Calculate exchange balance change
            indicators['exchange_balance_change'] = total_inflow - total_outflow
        
        # Analyze smart money wallets
        if self.smart_money_wallets:
            bullish_count = 0
            bearish_count = 0
            total_weight = 0
            
            for address, wallet in self.smart_money_wallets.items():
                if not wallet.get('history'):
                    continue
                
                recent_activity = wallet['history'][-1].get('activity', {})
                weight = wallet.get('weight', 5)
                total_weight += weight
                
                if recent_activity.get('action') == 'buy':
                    bullish_count += weight
                elif recent_activity.get('action') == 'sell':
                    bearish_count += weight
            
            if total_weight > 0:
                indicators['smart_money_sentiment'] = (bullish_count - bearish_count) / total_weight
                
                # Classify sentiment
                if indicators['smart_money_sentiment'] > 0.3:
                    indicators['smart_money_signal'] = 'bullish'
                elif indicators['smart_money_sentiment'] < -0.3:
                    indicators['smart_money_signal'] = 'bearish'
                else:
                    indicators['smart_money_signal'] = 'neutral'
        
        # Analyze network metrics
        if self.network_metrics:
            # Calculate transaction growth
            current_tx_count = self.network_metrics.get('transaction_count', 0)
            avg_tx_count = self.network_metrics.get('avg_transaction_count', 1)
            
            indicators['transaction_growth'] = (current_tx_count - avg_tx_count) / avg_tx_count if avg_tx_count > 0 else 0
            
            # Calculate active addresses growth
            current_active = self.network_metrics.get('active_addresses', 0)
            avg_active = self.network_metrics.get('avg_active_addresses', 1)
            
            indicators['active_addresses_growth'] = (current_active - avg_active) / avg_active if avg_active > 0 else 0
            
            # Calculate network utilization
            capacity = self.network_metrics.get('max_capacity', 1)
            current_usage = self.network_metrics.get('current_usage', 0)
            
            indicators['network_utilization'] = current_usage / capacity if capacity > 0 else 0
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on on-chain indicators.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            'buy': False,
            'sell': False,
            'entry_price': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'confidence': 0.0,
            'reasons': []
        }
        
        bullish_signals = 0
        bearish_signals = 0
        
        # Check whale flow signal
        if 'whale_net_flow_ratio' in indicators:
            whale_ratio = indicators['whale_net_flow_ratio']
            if whale_ratio > 1.5:
                bullish_signals += 1
                signals['reasons'].append(f"Whales accumulating (ratio: {whale_ratio:.2f})")
            elif whale_ratio < 0.7:
                bearish_signals += 1
                signals['reasons'].append(f"Whales distributing (ratio: {whale_ratio:.2f})")
        
        # Check exchange flow signal
        if 'exchange_flow_ratio' in indicators:
            flow_ratio = indicators['exchange_flow_ratio']
            threshold = self.parameters.get('exchange_inflow_outflow_ratio_threshold', 1.5)
            
            if flow_ratio > threshold:
                bearish_signals += 1
                signals['reasons'].append(f"High exchange inflows (ratio: {flow_ratio:.2f})")
            elif flow_ratio < 1 / threshold:
                bullish_signals += 1
                signals['reasons'].append(f"High exchange outflows (ratio: {flow_ratio:.2f})")
        
        # Check smart money signal
        if 'smart_money_signal' in indicators:
            signal = indicators['smart_money_signal']
            threshold = self.parameters.get('smart_money_follow_threshold', 0.7)
            
            if signal == 'bullish' and indicators.get('smart_money_sentiment', 0) > threshold:
                bullish_signals += 1.5  # Weighted more heavily
                signals['reasons'].append(f"Smart money bullish ({indicators.get('smart_money_sentiment', 0):.2f})")
            elif signal == 'bearish' and indicators.get('smart_money_sentiment', 0) < -threshold:
                bearish_signals += 1.5  # Weighted more heavily
                signals['reasons'].append(f"Smart money bearish ({indicators.get('smart_money_sentiment', 0):.2f})")
        
        # Check network metrics
        if 'transaction_growth' in indicators and 'active_addresses_growth' in indicators:
            tx_growth = indicators['transaction_growth']
            addr_growth = indicators['active_addresses_growth']
            
            if tx_growth > 0.2 and addr_growth > 0.1:
                bullish_signals += 1
                signals['reasons'].append(f"Growing network activity (tx: +{tx_growth:.1%}, addr: +{addr_growth:.1%})")
            elif tx_growth < -0.2 and addr_growth < -0.1:
                bearish_signals += 1
                signals['reasons'].append(f"Declining network activity (tx: {tx_growth:.1%}, addr: {addr_growth:.1%})")
        
        # Check price momentum alignment
        if 'price_momentum' in indicators:
            momentum = indicators['price_momentum']
            
            # If on-chain is bullish, check if price momentum aligns
            if bullish_signals > bearish_signals and momentum > 0:
                bullish_signals += 0.5
                signals['reasons'].append(f"Price momentum confirms bullish on-chain ({momentum:.1f}%)")
            elif bearish_signals > bullish_signals and momentum < 0:
                bearish_signals += 0.5
                signals['reasons'].append(f"Price momentum confirms bearish on-chain ({momentum:.1f}%)")
        
        # Determine signal confidence
        total_signals = bullish_signals + bearish_signals
        min_signals_required = self.parameters.get('signal_confirmation_count', 2)
        
        if total_signals > 0:
            if bullish_signals > bearish_signals and bullish_signals >= min_signals_required:
                signals['buy'] = True
                signals['confidence'] = bullish_signals / (total_signals * 0.5)
            elif bearish_signals > bullish_signals and bearish_signals >= min_signals_required:
                signals['sell'] = True
                signals['confidence'] = bearish_signals / (total_signals * 0.5)
        
        # Set entry price, stop loss and take profit
        current_price = data['close'].iloc[-1]
        
        signals['entry_price'] = current_price
        
        stop_loss_pct = self.parameters.get('stop_loss_pct', 5.0) / 100
        take_profit_pct = self.parameters.get('take_profit_pct', 10.0) / 100
        
        if signals['buy']:
            signals['stop_loss'] = current_price * (1 - stop_loss_pct)
            signals['take_profit'] = current_price * (1 + take_profit_pct)
        elif signals['sell']:
            signals['stop_loss'] = current_price * (1 + stop_loss_pct)
            signals['take_profit'] = current_price * (1 - take_profit_pct)
        
        # Track signal history
        self._track_signal(signals)
        
        return signals
    
    def _track_signal(self, signal: Dict[str, Any]) -> None:
        """
        Track signal history for performance analysis.
        
        Args:
            signal: Generated signal
        """
        # Store signal with timestamp
        signal_record = {
            'timestamp': datetime.now(timezone.utc),
            'buy': signal.get('buy', False),
            'sell': signal.get('sell', False),
            'confidence': signal.get('confidence', 0.0),
            'reasons': signal.get('reasons', [])
        }
        
        self.signal_history.append(signal_record)
        
        # Limit history size
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size based on on-chain signal confidence.
        
        Args:
            direction: Trade direction ('long' or 'short')
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size in crypto units
        """
        # Get signals
        signals = self.generate_signals(data, indicators)
        
        # Skip if signals don't match direction
        if (direction == 'long' and not signals.get('buy', False)) or \
           (direction == 'short' and not signals.get('sell', False)):
            return 0.0
        
        # Get account balance
        portfolio_value = self._calculate_portfolio_value()
        
        # Base position size on parameter
        base_position_pct = self.parameters.get('position_size_pct', 2.0) / 100
        
        # Adjust based on signal confidence
        confidence = signals.get('confidence', 0.0)
        adjusted_position_pct = base_position_pct * min(confidence, 1.0)
        
        # Ensure we don't exceed max concentration
        max_concentration_pct = self.parameters.get('max_concentration_pct', 15.0) / 100
        adjusted_position_pct = min(adjusted_position_pct, max_concentration_pct)
        
        # Calculate position size in fiat
        position_size_fiat = portfolio_value * adjusted_position_pct
        
        # Convert to crypto units
        current_price = data['close'].iloc[-1]
        position_size = position_size_fiat / current_price
        
        # Round to appropriate precision
        decimals = 8 if self.session.symbol.startswith('BTC') else 6
        position_size = round(position_size, decimals)
        
        logger.info(f"Calculated position size: {position_size} units (confidence: {confidence:.2f})")
        
        return position_size
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible this strategy is with the current market regime.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            "trending": 0.85,        # Good in trending markets
            "ranging": 0.65,         # Moderate in ranging markets
            "volatile": 0.70,        # Decent in volatile markets
            "calm": 0.60,            # Fair in calm markets
            "breakout": 0.90,        # Excellent for breakouts (can detect early)
            "high_volume": 0.85,     # Very good in high volume
            "low_volume": 0.60,      # Fair in low volume
            "high_liquidity": 0.80,  # Good in high liquidity markets
            "low_liquidity": 0.70,   # Decent in low liquidity markets
        }
        
        return compatibility_map.get(market_regime, 0.70)
