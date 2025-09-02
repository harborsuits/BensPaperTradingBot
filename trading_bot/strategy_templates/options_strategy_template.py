#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Options Strategy Template for BensBot Trading System

This template defines the standard structure that all options strategies should follow
to ensure proper integration with the backtester, strategy finder, and live trading.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd

from trading_bot.strategy_templates.strategy_template import StrategyTemplate, register_strategy_with_registry
from trading_bot.market.market_data import MarketData
from trading_bot.market.universe import Universe
from trading_bot.market.option_chains import OptionChains

logger = logging.getLogger(__name__)

class OptionsStrategyTemplate(StrategyTemplate):
    """
    Base template for all options trading strategies that ensures consistent
    interface for the backtester, strategy finder, and live trading systems.
    
    All options strategies should inherit from this template and implement the 
    required methods with the same signatures.
    """
    
    # Default parameters for options strategies
    DEFAULT_PARAMS = {
        'strategy_name': 'options_template_strategy',
        'strategy_version': '1.0.0',
        'asset_class': 'options',
        'strategy_type': 'all_weather',
        'timeframe': 'daily',
        'market_regime': 'all_weather',
        
        # Options-specific parameters
        'min_stock_price': 30.0,              # Minimum stock price to consider
        'max_stock_price': 500.0,             # Maximum stock price to consider
        'min_option_volume': 50,              # Minimum option volume
        'min_option_open_interest': 100,      # Minimum option open interest
        'min_iv_percentile': 30,              # Minimum IV percentile
        'max_iv_percentile': 60,              # Maximum IV percentile
        
        # Expiration parameters
        'target_dte': 45,                     # Target days to expiration
        'min_dte': 30,                        # Minimum days to expiration
        'max_dte': 60,                        # Maximum days to expiration
        
        # Risk management parameters
        'max_position_size_percent': 0.05,    # Maximum position size as % of portfolio
        'max_num_positions': 10,              # Maximum number of positions
        'max_risk_per_trade': 0.02,           # Maximum risk per trade as % of portfolio
    }
    
    def __init__(self, 
                 strategy_id: str = None, 
                 name: str = None,
                 parameters: Dict[str, Any] = None):
        """
        Initialize the options strategy with parameters.
        
        Args:
            strategy_id: Unique identifier for this strategy instance
            name: Human-readable name of the strategy
            parameters: Strategy-specific parameters that override the defaults
        """
        super().__init__(strategy_id, name, parameters)
        
        # Options-specific tracking
        self.option_positions = {}            # Track option positions by contract
        self.underlying_positions = {}        # Track underlying stock positions
        self.option_chains_cache = {}         # Cache recent option chain data
        self.iv_history = {}                  # Track IV history by symbol
        
    def define_universe(self, market_data: MarketData) -> Universe:
        """
        Define the universe of tradable assets for this options strategy.
        
        Args:
            market_data: Market data to use for filtering
            
        Returns:
            Universe object containing filtered symbols
        """
        # Default universe definition for options strategies
        universe = Universe()
        
        # Get basic filtering parameters
        min_price = self.parameters.get('min_stock_price', 30.0)
        max_price = self.parameters.get('max_stock_price', 500.0)
        
        # Filter by price, volume, and has options
        all_symbols = market_data.get_all_symbols()
        filtered_symbols = []
        
        for symbol in all_symbols:
            quote = market_data.get_latest_quote(symbol)
            if not quote:
                continue
                
            price = quote.get('price')
            volume = quote.get('volume', 0)
            
            # Basic price and volume filters
            if price and min_price <= price <= max_price and volume >= 100000:
                # Check if options are available for this symbol
                if market_data.has_options(symbol):
                    filtered_symbols.append(symbol)
        
        universe.add_symbols(filtered_symbols)
        self.logger.info(f"Options universe defined with {len(filtered_symbols)} symbols")
        return universe
    
    def generate_signals(self, market_data: Union[MarketData, Dict[str, Any]], 
                        option_chains: Optional[OptionChains] = None) -> List[Dict[str, Any]]:
        """
        Generate options trading signals based on market data and option chains.
        
        Args:
            market_data: Market data for analysis
            option_chains: Option chain data for the symbols
            
        Returns:
            List of signal dictionaries with standard format
        """
        # This method should be implemented by specific options strategies
        # Default implementation returns empty list
        return []
    
    def _analyze_option_chains(self, symbol: str, chains: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze option chains for a symbol to find tradable contracts.
        
        Args:
            symbol: Stock symbol
            chains: Option chain data
            
        Returns:
            List of tradable option contracts
        """
        # Default option chain analysis
        tradable_options = []
        
        # Extract parameters
        min_dte = self.parameters.get('min_dte')
        max_dte = self.parameters.get('max_dte')
        min_volume = self.parameters.get('min_option_volume')
        min_oi = self.parameters.get('min_option_open_interest')
        
        # Get current stock price
        stock_price = chains.get('underlying_price')
        if not stock_price:
            return []
            
        # Analyze call options
        for option in chains.get('calls', []):
            expiration = option.get('expiration')
            if not expiration:
                continue
                
            # Calculate days to expiration
            expiry_date = datetime.strptime(expiration, '%Y-%m-%d').date()
            today = date.today()
            dte = (expiry_date - today).days
            
            # Apply DTE filter
            if dte < min_dte or dte > max_dte:
                continue
                
            # Apply volume and open interest filters
            volume = option.get('volume', 0)
            open_interest = option.get('open_interest', 0)
            if volume < min_volume or open_interest < min_oi:
                continue
                
            # Contract passes initial screening
            tradable_options.append(option)
        
        # Repeat for put options
        for option in chains.get('puts', []):
            expiration = option.get('expiration')
            if not expiration:
                continue
                
            # Calculate days to expiration
            expiry_date = datetime.strptime(expiration, '%Y-%m-%d').date()
            today = date.today()
            dte = (expiry_date - today).days
            
            # Apply DTE filter
            if dte < min_dte or dte > max_dte:
                continue
                
            # Apply volume and open interest filters
            volume = option.get('volume', 0)
            open_interest = option.get('open_interest', 0)
            if volume < min_volume or open_interest < min_oi:
                continue
                
            # Contract passes initial screening
            tradable_options.append(option)
        
        return tradable_options
    
    def position_sizing(self, signal: Dict[str, Any], account_info: Dict[str, Any]) -> float:
        """
        Calculate position size for a given option signal.
        
        Args:
            signal: Trading signal dictionary
            account_info: Account information including equity, margin, etc.
            
        Returns:
            Number of option contracts to trade
        """
        # Default options position sizing
        account_value = account_info.get('equity', 0)
        if account_value <= 0:
            return 0
            
        # Extract parameters
        max_position_pct = self.parameters.get('max_position_size_percent', 0.05)
        max_risk_pct = self.parameters.get('max_risk_per_trade', 0.02)
        
        # Maximum amount to allocate to this trade
        max_amount = account_value * max_position_pct
        
        # Calculate position size based on premium
        premium = signal.get('premium', 0)
        if premium <= 0:
            return 0
            
        # Each option contract is for 100 shares
        contract_cost = premium * 100
        
        # Calculate how many contracts we can afford
        max_contracts = int(max_amount / contract_cost)
        
        # Calculate position size based on risk
        max_risk_amount = account_value * max_risk_pct
        stop_loss = signal.get('stop_loss')
        entry_price = signal.get('entry_price')
        
        if stop_loss and entry_price:
            risk_per_contract = abs(entry_price - stop_loss) * 100
            if risk_per_contract > 0:
                risk_based_contracts = int(max_risk_amount / risk_per_contract)
                # Take the smaller of the two limits
                max_contracts = min(max_contracts, risk_based_contracts)
        
        return max_contracts
    
    def calculate_risk_metrics(self, signal: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate risk metrics for an options trade.
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            Dictionary of risk metrics
        """
        # Options-specific risk metrics
        action = signal.get('action', '')
        option_type = signal.get('option_type', '')
        strike = signal.get('strike', 0)
        premium = signal.get('premium', 0)
        stock_price = signal.get('stock_price', 0)
        
        max_loss = 0
        max_profit = 0
        risk_reward = 0
        breakeven = 0
        
        # Buying call
        if action == 'BUY' and option_type == 'CALL':
            max_loss = premium
            max_profit = float('inf')  # Theoretically unlimited
            breakeven = strike + premium
            risk_reward = float('inf')  # Theoretically unlimited
            
        # Buying put
        elif action == 'BUY' and option_type == 'PUT':
            max_loss = premium
            max_profit = strike - premium  # Max is if stock goes to zero
            breakeven = strike - premium
            if max_loss > 0:
                risk_reward = max_profit / max_loss
                
        # Selling call
        elif action == 'SELL' and option_type == 'CALL':
            max_profit = premium
            max_loss = float('inf')  # Theoretically unlimited
            breakeven = strike + premium
            risk_reward = 0  # Undefined for unlimited risk
            
        # Selling put
        elif action == 'SELL' and option_type == 'PUT':
            max_profit = premium
            max_loss = strike - premium  # Max is if stock goes to zero
            breakeven = strike - premium
            if max_loss > 0:
                risk_reward = max_profit / max_loss
        
        return {
            'max_loss': max_loss,
            'max_profit': max_profit,
            'risk_reward_ratio': risk_reward,
            'breakeven': breakeven,
            'premium': premium,
        }
    
    # Helper methods for options strategies
    def calculate_iv_percentile(self, symbol: str, current_iv: float) -> float:
        """
        Calculate the IV percentile for a symbol.
        
        Args:
            symbol: Stock symbol
            current_iv: Current implied volatility
            
        Returns:
            IV percentile (0-100)
        """
        # Get historical IV data from cache or fetch it
        iv_history = self.iv_history.get(symbol, [])
        if not iv_history:
            # This would normally fetch historical IV data
            return 50  # Default to middle percentile
            
        # Calculate percentile
        if iv_history:
            return np.percentile(iv_history.index(current_iv) / len(iv_history) * 100, 0)
        return 50
    
    def create_options_signal(self, symbol: str, action: str, option_type: str,
                            strike: float, expiration: str, premium: float,
                            stock_price: float, reason: str, strength: float = 1.0,
                            stop_loss: Optional[float] = None, 
                            take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a standardized options signal dictionary.
        
        Args:
            symbol: Underlying stock symbol
            action: Signal action ('BUY', 'SELL')
            option_type: Option type ('CALL', 'PUT')
            strike: Strike price
            expiration: Expiration date
            premium: Option premium
            stock_price: Current stock price
            reason: Reason for the signal
            strength: Signal strength from 0.0 to 1.0
            stop_loss: Suggested stop loss price
            take_profit: Suggested take profit price
            
        Returns:
            Standardized options signal dictionary
        """
        # Create base signal
        signal = self.create_signal(
            symbol=symbol,
            action=action,
            reason=reason,
            strength=strength,
            entry_price=premium,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Add options-specific fields
        signal.update({
            'option_type': option_type,
            'strike': strike,
            'expiration': expiration,
            'premium': premium,
            'stock_price': stock_price,
            'contract_multiplier': 100,  # Standard for equity options
        })
        
        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(signal)
        signal.update(risk_metrics)
        
        return signal
