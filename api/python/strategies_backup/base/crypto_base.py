#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Base Strategy Module

This module provides the base class for cryptocurrency trading strategies, with
crypto-specific functionality built in.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum

from trading_bot.strategies.strategy_template import StrategyOptimizable, Signal, SignalType, TimeFrame, MarketRegime

logger = logging.getLogger(__name__)

class ExchangeType(Enum):
    """Enum for cryptocurrency exchanges."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    FTX = "ftx"
    BITFINEX = "bitfinex"
    KUCOIN = "kucoin"

class CryptoBaseStrategy(StrategyOptimizable):
    """
    Base class for cryptocurrency trading strategies.
    
    This class extends the StrategyOptimizable to add crypto-specific
    functionality including:
    - Handling 24/7 markets
    - Exchange-specific operations
    - On-chain data integration
    - Crypto-specific technical indicators
    - High volatility handling
    - Support for stablecoins and trading pairs
    """
    
    # Default parameters specific to crypto trading
    DEFAULT_CRYPTO_PARAMS = {
        # Market data parameters
        'trading_hours': 24,              # Crypto markets are 24/7
        'min_price': 0.0,                 # No minimum price (some are fractions of cents)
        'min_volume_usd': 1000000,        # Minimum 24h volume in USD
        'min_market_cap_usd': 10000000,   # Minimum market cap $10M
        
        # Exchange parameters
        'exchange': ExchangeType.BINANCE, # Default exchange
        'quote_currency': 'USDT',         # Quote currency for trading pairs
        'min_trade_size': 10.0,           # Minimum trade size in USD
        'maker_fee': 0.001,               # Maker fee (0.1%)
        'taker_fee': 0.001,               # Taker fee (0.1%)
        
        # Volatility parameters
        'max_volatility': 0.10,           # Maximum 24h volatility (10%)
        'volatility_adjustment': True,    # Adjust position size for volatility
        
        # Technical parameters
        'use_funding_rate': False,        # Whether to use funding rate analysis
        'use_order_book': False,          # Whether to use order book analysis
        'use_onchain_data': False,        # Whether to use on-chain analytics
        
        # Risk parameters
        'max_position_size_percent': 0.05, # Maximum position size (5%)
        'stablecoin_allocation': 0.30,     # Allocation to stablecoins
        'max_drawdown_exit': 0.15,         # Exit if drawdown exceeds 15%
        'trailing_stop_percent': 0.05,     # Trailing stop (5%)
        
        # Crypto-specific parameters
        'btc_correlation_threshold': 0.7,  # BTC correlation threshold
        'use_defi_metrics': False,         # Whether to use DeFi-specific metrics
        'focus_on_large_caps': True,       # Focus on larger market cap coins
    }
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a cryptocurrency trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters (will be merged with DEFAULT_CRYPTO_PARAMS)
            metadata: Strategy metadata
        """
        # Start with default crypto parameters
        crypto_params = self.DEFAULT_CRYPTO_PARAMS.copy()
        
        # Override with provided parameters
        if parameters:
            crypto_params.update(parameters)
        
        # Initialize the parent class
        super().__init__(name=name, parameters=crypto_params, metadata=metadata)
        
        # Crypto-specific member variables
        self.market_caps = {}  # Track market caps for allocation weighting
        self.btc_correlation = {}  # Track correlation with BTC
        
        logger.info(f"Initialized crypto strategy: {name}")
    
    def filter_universe(self, universe: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Filter the universe based on crypto-specific criteria.
        
        Args:
            universe: Dictionary mapping symbols to DataFrames with crypto data
            
        Returns:
            Filtered universe
        """
        filtered_universe = {}
        
        for symbol, data in universe.items():
            # Skip if no data
            if data.empty:
                continue
            
            # Skip stablecoins if not explicitly allowed
            if self._is_stablecoin(symbol) and not self.parameters.get('include_stablecoins', False):
                continue
            
            # Skip if below minimum volume
            if 'volume_usd_24h' in data.columns and self.parameters['min_volume_usd'] > 0:
                latest_volume = data['volume_usd_24h'].iloc[-1]
                if latest_volume < self.parameters['min_volume_usd']:
                    continue
            
            # Skip if below minimum market cap
            if 'market_cap_usd' in data.columns and self.parameters['min_market_cap_usd'] > 0:
                latest_mcap = data['market_cap_usd'].iloc[-1]
                if latest_mcap < self.parameters['min_market_cap_usd']:
                    continue
            
            # Skip if volatility too high
            if 'volatility_24h' in data.columns and self.parameters['max_volatility'] > 0:
                latest_vol = data['volatility_24h'].iloc[-1]
                if latest_vol > self.parameters['max_volatility']:
                    continue
            
            # Symbol passed all filters
            filtered_universe[symbol] = data
        
        logger.info(f"Filtered crypto universe from {len(universe)} to {len(filtered_universe)} symbols")
        return filtered_universe
    
    def calculate_crypto_indicators(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate crypto-specific technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Calculate Moving Averages
        for period in [20, 50, 200]:
            ma_key = f'ma_{period}'
            indicators[ma_key] = pd.DataFrame({
                ma_key: data['close'].rolling(window=period).mean()
            })
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = pd.DataFrame({'rsi': rsi})
        
        # Calculate MACD
        ema12 = data['close'].ewm(span=12, adjust=False).mean()
        ema26 = data['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        
        indicators['macd'] = pd.DataFrame({
            'macd_line': macd_line,
            'signal_line': signal_line,
            'macd_hist': macd_hist
        })
        
        # Calculate Bollinger Bands
        ma20 = data['close'].rolling(window=20).mean()
        std20 = data['close'].rolling(window=20).std()
        
        upper_band = ma20 + (std20 * 2)
        lower_band = ma20 - (std20 * 2)
        
        indicators['bbands'] = pd.DataFrame({
            'middle_band': ma20,
            'upper_band': upper_band,
            'lower_band': lower_band
        })
        
        # Crypto-specific: Calculate VWAP (Volume Weighted Average Price)
        if 'volume' in data.columns:
            # Reset index to use datetime for calculations
            if isinstance(data.index, pd.DatetimeIndex):
                tmp_data = data.copy()
            else:
                tmp_data = data.set_index(pd.DatetimeIndex(data['timestamp']) if 'timestamp' in data.columns else None)
            
            # Group by trading day (UTC) for VWAP calculation
            tmp_data['date'] = tmp_data.index.date
            
            # Calculate typical price
            tmp_data['typical_price'] = (tmp_data['high'] + tmp_data['low'] + tmp_data['close']) / 3
            
            # Calculate VWAP components
            tmp_data['vwap_numerator'] = tmp_data['typical_price'] * tmp_data['volume']
            tmp_data['vwap_denominator'] = tmp_data['volume']
            
            # Group by date and calculate cumulative sums
            grouped = tmp_data.groupby('date')
            vwap_num_cs = grouped['vwap_numerator'].cumsum()
            vwap_den_cs = grouped['vwap_denominator'].cumsum()
            
            # Calculate VWAP
            vwap = vwap_num_cs / vwap_den_cs
            
            # Store in indicators
            indicators['vwap'] = pd.DataFrame({'vwap': vwap})
        
        # Add on-chain data indicators if enabled
        if self.parameters['use_onchain_data'] and 'active_addresses' in data.columns:
            # Create Network Value to Transactions (NVT) ratio
            # NVT = Market Cap / Transaction Volume
            if 'market_cap_usd' in data.columns and 'transaction_volume_usd' in data.columns:
                nvt = data['market_cap_usd'] / data['transaction_volume_usd']
                indicators['nvt'] = pd.DataFrame({'nvt': nvt})
            
            # Active Addresses Ratio (AAR)
            # AAR = Active Addresses / Total Addresses
            if 'active_addresses' in data.columns and 'total_addresses' in data.columns:
                aar = data['active_addresses'] / data['total_addresses']
                indicators['aar'] = pd.DataFrame({'aar': aar})
            
            # Average Transaction Value (ATV)
            if 'transaction_volume_usd' in data.columns and 'transaction_count' in data.columns:
                atv = data['transaction_volume_usd'] / data['transaction_count']
                indicators['atv'] = pd.DataFrame({'atv': atv})
        
        return indicators
    
    def calculate_position_size(self, symbol: str, price: float, volatility: float, 
                              risk_amount: float) -> float:
        """
        Calculate position size accounting for crypto-specific factors.
        
        Args:
            symbol: Crypto symbol
            price: Current price
            volatility: Volatility (daily standard deviation)
            risk_amount: Amount to risk in USD
            
        Returns:
            Position size in units of the cryptocurrency
        """
        # Base position size from risk amount
        base_position_size = risk_amount / price
        
        # Apply volatility adjustment if enabled
        if self.parameters['volatility_adjustment'] and volatility > 0:
            # Adjust size inversely with volatility (more volatile = smaller size)
            # Normalize by a 5% daily volatility baseline
            volatility_factor = 0.05 / volatility
            adjusted_position = base_position_size * min(volatility_factor, 1.0)
        else:
            adjusted_position = base_position_size
        
        # Apply market cap scaling if available
        if symbol in self.market_caps and self.parameters['focus_on_large_caps']:
            # Larger market cap = larger position
            market_cap = self.market_caps[symbol]
            # Scale by log market cap difference from $1B
            log_mcap_factor = np.log10(market_cap) / np.log10(1e9)
            mcap_adjusted_position = adjusted_position * min(log_mcap_factor, 1.5)
        else:
            mcap_adjusted_position = adjusted_position
        
        # Apply BTC correlation factor if available
        if symbol in self.btc_correlation and self.parameters['btc_correlation_threshold'] > 0:
            btc_corr = self.btc_correlation[symbol]
            # If correlation is higher than threshold, reduce position
            if abs(btc_corr) > self.parameters['btc_correlation_threshold']:
                correlation_factor = 1.0 - (abs(btc_corr) - self.parameters['btc_correlation_threshold'])
                final_position = mcap_adjusted_position * max(correlation_factor, 0.5)
            else:
                final_position = mcap_adjusted_position
        else:
            final_position = mcap_adjusted_position
        
        # Ensure minimum trade size
        min_trade_size_in_units = self.parameters['min_trade_size'] / price
        
        if final_position < min_trade_size_in_units:
            return 0.0  # Too small to trade
        
        return final_position
    
    def adjust_for_market_hours(self, signals: Dict[str, Signal]) -> Dict[str, Signal]:
        """
        Crypto markets are 24/7, but activity varies by time. This adjusts signals
        based on typical activity patterns by time of day.
        
        Args:
            signals: Dictionary of generated signals
            
        Returns:
            Adjusted signals
        """
        # Skip if not enabled in parameters
        if not self.parameters.get('adjust_for_market_hours', False):
            return signals
            
        adjusted_signals = signals.copy()
        current_hour = datetime.utcnow().hour
        
        # Define high activity hours (UTC)
        high_activity_hours = {9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22}
        low_activity_hours = {0, 1, 2, 3, 4, 5}
        
        for symbol, signal in adjusted_signals.items():
            # During high activity, increase confidence
            if current_hour in high_activity_hours:
                signal.confidence = min(1.0, signal.confidence * 1.2)
                
            # During low activity, decrease confidence
            elif current_hour in low_activity_hours:
                signal.confidence = signal.confidence * 0.8
        
        return adjusted_signals
    
    def analyze_onchain_data(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze on-chain data metrics if available.
        
        Args:
            symbol: Crypto symbol
            data: DataFrame with on-chain metrics
            
        Returns:
            Dictionary of on-chain analysis results
        """
        # Skip if not enabled or data not available
        if not self.parameters['use_onchain_data'] or data.empty:
            return {}
            
        analysis = {}
        
        # Check if all required columns exist
        required_columns = ['active_addresses', 'transaction_count', 'transaction_volume_usd']
        if not all(col in data.columns for col in required_columns):
            return {}
            
        # Get latest data point
        latest = data.iloc[-1]
        
        # Calculate NVT ratio (Network Value to Transactions)
        if 'market_cap_usd' in data.columns:
            nvt = latest['market_cap_usd'] / latest['transaction_volume_usd']
            analysis['nvt'] = nvt
            
            # Interpret NVT (lower is generally more bullish)
            if nvt < 15:
                analysis['nvt_signal'] = 1.0  # Very bullish
            elif nvt < 30:
                analysis['nvt_signal'] = 0.5  # Moderately bullish
            elif nvt > 100:
                analysis['nvt_signal'] = -1.0  # Very bearish
            elif nvt > 50:
                analysis['nvt_signal'] = -0.5  # Moderately bearish
            else:
                analysis['nvt_signal'] = 0.0  # Neutral
        
        # Check address growth
        if len(data) > 30 and 'active_addresses' in data.columns:
            addr_growth_30d = latest['active_addresses'] / data['active_addresses'].iloc[-30] - 1
            analysis['address_growth_30d'] = addr_growth_30d
            
            # Interpret address growth
            if addr_growth_30d > 0.2:  # 20% monthly growth
                analysis['address_signal'] = 1.0  # Very bullish
            elif addr_growth_30d > 0.05:  # 5% monthly growth
                analysis['address_signal'] = 0.5  # Moderately bullish
            elif addr_growth_30d < -0.1:  # 10% monthly decline
                analysis['address_signal'] = -1.0  # Very bearish
            elif addr_growth_30d < -0.02:  # 2% monthly decline
                analysis['address_signal'] = -0.5  # Moderately bearish
            else:
                analysis['address_signal'] = 0.0  # Neutral
        
        # Average confidence from all signals
        if any(key.endswith('_signal') for key in analysis):
            signals = [val for key, val in analysis.items() if key.endswith('_signal')]
            analysis['overall_onchain_sentiment'] = sum(signals) / len(signals)
        
        return analysis
    
    def _is_stablecoin(self, symbol: str) -> bool:
        """
        Check if a symbol is a stablecoin.
        
        Args:
            symbol: Crypto symbol
            
        Returns:
            True if symbol is a stablecoin
        """
        # List of common stablecoins
        stablecoins = [
            'USDT', 'USDC', 'DAI', 'BUSD', 'UST', 'TUSD', 'USDP', 'GUSD',
            'USDT-USD', 'USDC-USD', 'DAI-USD', 'BUSD-USD'
        ]
        
        # Check if symbol matches any stablecoin
        return symbol in stablecoins or any(symbol.startswith(s + "/") for s in stablecoins) 