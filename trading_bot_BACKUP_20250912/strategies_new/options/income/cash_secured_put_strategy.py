#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cash-Secured Put Strategy

This module implements a cash-secured put options strategy which involves selling put options
while maintaining enough cash to purchase the underlying shares if the option is exercised.
The strategy is account-aware, ensuring it complies with account balance requirements,
regulatory constraints, and position sizing.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from trading_bot.strategies_new.options.base.options_base_strategy import OptionsBaseStrategy, OptionsSession
from trading_bot.strategies_new.validation.account_aware_mixin import AccountAwareMixin
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.position_management.position import Position
from trading_bot.strategies_new.factory.registry import register_strategy

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="CashSecuredPutStrategy",
    market_type="options",
    description="A strategy that involves selling put options while maintaining enough cash to purchase the underlying shares if the option is exercised",
    timeframes=["1d", "1w"],
    parameters={
        "delta_target": {"description": "Target delta for put selection", "type": "float"},
        "days_to_expiration_min": {"description": "Minimum days to expiration", "type": "int"},
        "days_to_expiration_max": {"description": "Maximum days to expiration", "type": "int"},
        "profit_target_pct": {"description": "Take profit at this percentage of max premium", "type": "float"}
    }
)
class CashSecuredPutStrategy(OptionsBaseStrategy, AccountAwareMixin):
    """
    Cash-Secured Put Options Strategy
    
    This strategy:
    1. Identifies quality stocks at reasonable valuations to sell puts on
    2. Sells out-of-the-money put options while maintaining enough cash for assignment
    3. Collects premium income while potentially acquiring stocks at a discount
    4. Integrates account awareness to verify sufficient collateral is available
    5. Manages positions with proper risk management and exit criteria
    
    The strategy is primarily income-focused but also used for potential stock acquisition
    at prices below current market value.
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the Cash-Secured Put strategy.
        
        Args:
            session: Options trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Initialize parent classes
        OptionsBaseStrategy.__init__(self, session, data_pipeline, parameters)
        AccountAwareMixin.__init__(self)
        
        # Default parameters for cash-secured puts
        default_params = {
            # Strategy identification
            'strategy_name': 'Cash-Secured Put',
            'strategy_id': 'cash_secured_put',
            'is_day_trade': False,  # This is not a day trading strategy
            
            # Stock selection parameters
            'min_stock_price': 20.0,  # Minimum stock price
            'max_stock_price': 500.0,  # Maximum stock price
            'min_market_cap': 1000000000,  # $1B minimum market cap
            'min_avg_volume': 500000,  # 500K minimum average volume
            
            # Option selection parameters
            'delta_target': 0.30,  # Target delta for put selection
            'delta_range': 0.05,  # Acceptable delta range (+/-)
            'days_to_expiration_min': 30,  # Minimum DTE
            'days_to_expiration_max': 45,  # Maximum DTE
            'min_premium_percent': 0.01,  # Minimum premium (1% of strike)
            'min_open_interest': 50,  # Minimum open interest
            'min_option_volume': 10,  # Minimum daily option volume
            
            # Exit parameters
            'profit_target_pct': 0.50,  # Close at 50% of max profit
            'days_to_close': 7,  # Close position when less than 7 days to expiry
            'max_loss_percent': 0.50,  # Maximum acceptable loss (% of premium)
            
            # Management parameters
            'roll_when_tested': True,  # Roll when strike price is tested
            'roll_days_threshold': 7,  # Roll when this close to expiration if tested
            'roll_to_delta': 0.30,  # Target delta for rolls
            'max_positions': 10,  # Maximum number of positions
            
            # Risk management
            'max_position_size_pct': 0.05,  # Maximum 5% of account in a single position
            'max_sector_exposure': 0.25,  # Maximum 25% exposure to a single sector
            'risk_per_trade': 0.01,  # 1% risk per trade
        }
        
        # Update with user-provided parameters
        if parameters:
            default_params.update(parameters)
        self.parameters = default_params
        
        # Strategy state
        self.current_positions = []  # Track open positions
        self.pending_rolls = []  # Track positions pending roll
        self.sector_exposure = {}  # Track sector exposure
        
        logger.info(f"Initialized {self.name} for {session.symbol} on {session.timeframe}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the Cash-Secured Put strategy.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if data.empty or len(data) < 20:  # Need at least 20 bars for indicators
            return indicators
        
        try:
            # Calculate trend indicators
            indicators['sma_20'] = data['close'].rolling(window=20).mean()
            indicators['sma_50'] = data['close'].rolling(window=50).mean()
            indicators['sma_200'] = data['close'].rolling(window=200).mean()
            
            # Price relative to moving averages
            indicators['price_vs_sma20'] = data['close'] / indicators['sma_20'] - 1
            indicators['price_vs_sma50'] = data['close'] / indicators['sma_50'] - 1
            indicators['price_vs_sma200'] = data['close'] / indicators['sma_200'] - 1
            
            # Calculate volatility indicators
            indicators['atr_14'] = self._calculate_atr(data, window=14)
            indicators['historical_volatility'] = data['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            
            # Calculate RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            indicators['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
            indicators['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
            indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']
            
            # Calculate Bollinger Bands
            indicators['bb_middle'] = data['close'].rolling(window=20).mean()
            indicators['bb_std'] = data['close'].rolling(window=20).std()
            indicators['bb_upper'] = indicators['bb_middle'] + (indicators['bb_std'] * 2)
            indicators['bb_lower'] = indicators['bb_middle'] - (indicators['bb_std'] * 2)
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
            
            # Volume indicators
            indicators['volume_sma'] = data['volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = data['volume'] / indicators['volume_sma']
            
            # Market trend strength
            indicators['trend_strength'] = self._calculate_trend_strength(data, indicators)
            
            # Option-specific indicators
            if 'implied_volatility' in data.columns:
                indicators['iv_percentile'] = self._calculate_iv_percentile(data['implied_volatility'])
                indicators['iv_rank'] = self._calculate_iv_rank(data['implied_volatility'])
            
        except Exception as e:
            logger.error(f"Error calculating Cash-Secured Put indicators: {str(e)}")
        
        return indicators
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: OHLCV DataFrame
            window: Look-back period
            
        Returns:
            ATR series
        """
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def _calculate_trend_strength(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> pd.Series:
        """
        Calculate a composite trend strength indicator.
        
        Args:
            data: OHLCV DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Trend strength series (0-100)
        """
        # Start with a neutral score
        trend_strength = pd.Series(50, index=data.index)
        
        # Price relative to moving averages
        if 'price_vs_sma20' in indicators and 'price_vs_sma50' in indicators and 'price_vs_sma200' in indicators:
            # Add points if price is above SMAs (bullish)
            sma20_factor = np.where(indicators['price_vs_sma20'] > 0, 10, -10)
            sma50_factor = np.where(indicators['price_vs_sma50'] > 0, 10, -10)
            sma200_factor = np.where(indicators['price_vs_sma200'] > 0, 10, -10)
            
            trend_strength += pd.Series(sma20_factor + sma50_factor + sma200_factor, index=data.index)
        
        # MACD histogram direction
        if 'macd_hist' in indicators:
            macd_factor = np.where(indicators['macd_hist'] > 0, 10, -10)
            trend_strength += pd.Series(macd_factor, index=data.index)
        
        # RSI
        if 'rsi_14' in indicators:
            rsi_factor = np.where(indicators['rsi_14'] > 50, 5, -5)
            trend_strength += pd.Series(rsi_factor, index=data.index)
        
        # Volume confirmation
        if 'volume_ratio' in indicators:
            volume_factor = np.where(indicators['volume_ratio'] > 1.2, 5, 0)
            trend_strength += pd.Series(volume_factor, index=data.index)
        
        # Clip values to 0-100 range
        return trend_strength.clip(0, 100)
    
    def _calculate_iv_percentile(self, iv_series: pd.Series, window: int = 252) -> float:
        """
        Calculate implied volatility percentile.
        
        Args:
            iv_series: Implied volatility series
            window: Look-back period
            
        Returns:
            IV percentile (0-100)
        """
        if len(iv_series) < 2:
            return 50.0  # Default to middle if not enough data
            
        # Get the most recent IV
        current_iv = iv_series.iloc[-1]
        
        # Calculate the percentile
        lookback = min(len(iv_series), window)
        iv_history = iv_series.iloc[-lookback:]
        
        # Percentile calculation
        pct = sum(1 for x in iv_history if x < current_iv) / len(iv_history) * 100
        
        return pct
    
    def _calculate_iv_rank(self, iv_series: pd.Series, window: int = 252) -> float:
        """
        Calculate implied volatility rank.
        
        Args:
            iv_series: Implied volatility series
            window: Look-back period
            
        Returns:
            IV rank (0-100)
        """
        if len(iv_series) < 2:
            return 50.0  # Default to middle if not enough data
            
        # Get the most recent IV
        current_iv = iv_series.iloc[-1]
        
        # Calculate the rank
        lookback = min(len(iv_series), window)
        iv_history = iv_series.iloc[-lookback:]
        
        iv_min = iv_history.min()
        iv_max = iv_history.max()
        
        # Avoid division by zero
        if iv_max == iv_min:
            return 50.0
            
        iv_rank = (current_iv - iv_min) / (iv_max - iv_min) * 100
        
        return iv_rank
    
    def select_option_contract(self, data: pd.DataFrame, option_chain: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Select the appropriate put option to sell based on our parameters.
        
        Args:
            data: Market data DataFrame
            option_chain: Option chain DataFrame
            
        Returns:
            Selected option contract or None
        """
        try:
            if data.empty or option_chain.empty:
                return None
                
            # Get the current underlying price
            current_price = data['close'].iloc[-1]
            
            # Filter for put options
            puts = option_chain[option_chain['option_type'] == 'put']
            
            if puts.empty:
                logger.warning("No put options available in the option chain")
                return None
                
            # Apply our selection criteria
            puts = puts[
                # Days to expiration within our range
                (puts['days_to_expiration'] >= self.parameters['days_to_expiration_min']) &
                (puts['days_to_expiration'] <= self.parameters['days_to_expiration_max']) &
                # Delta within our target range
                (puts['delta'].abs() >= self.parameters['delta_target'] - self.parameters['delta_range']) &
                (puts['delta'].abs() <= self.parameters['delta_target'] + self.parameters['delta_range']) &
                # Minimum liquidity
                (puts['open_interest'] >= self.parameters['min_open_interest']) &
                (puts['volume'] >= self.parameters['min_option_volume']) &
                # Strike below current price (OTM puts)
                (puts['strike'] < current_price)
            ]
            
            if puts.empty:
                logger.warning("No suitable put options found matching criteria")
                return None
            
            # Sort by delta (closest to our target)
            puts['delta_distance'] = abs(puts['delta'].abs() - self.parameters['delta_target'])
            puts = puts.sort_values('delta_distance')
            
            # Get the best candidate
            best_put = puts.iloc[0]
            
            # Calculate premium as percentage of strike
            premium_pct = best_put['bid'] / best_put['strike']
            
            # Check if premium meets our minimum
            if premium_pct < self.parameters['min_premium_percent']:
                logger.info(f"Best put option premium ({premium_pct:.2%}) below minimum threshold ({self.parameters['min_premium_percent']:.2%})")
                return None
            
            # Create contract details
            contract = {
                'symbol': best_put['symbol'],
                'option_type': 'put',
                'strike': best_put['strike'],
                'expiration': best_put['expiration'],
                'days_to_expiration': best_put['days_to_expiration'],
                'delta': best_put['delta'],
                'bid': best_put['bid'],
                'ask': best_put['ask'],
                'mid': (best_put['bid'] + best_put['ask']) / 2,
                'open_interest': best_put['open_interest'],
                'volume': best_put['volume'],
                'premium_pct': premium_pct
            }
            
            logger.info(f"Selected put option: Strike=${contract['strike']:.2f}, "
                       f"Expiry={contract['days_to_expiration']} days, "
                       f"Delta={contract['delta']:.2f}, "
                       f"Premium={premium_pct:.2%}")
            
            return contract
            
        except Exception as e:
            logger.error(f"Error selecting put option: {str(e)}")
            return None
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for the Cash-Secured Put strategy.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            'entry': False,
            'exit': False,
            'contract': None,
            'position_size': 0,
            'positions_to_close': []
        }
        
        if data.empty or not indicators:
            return signals
        
        try:
            current_price = data['close'].iloc[-1]
            
            # Check if price meets our stock selection criteria
            if current_price < self.parameters['min_stock_price'] or current_price > self.parameters['max_stock_price']:
                logger.debug(f"Stock price (${current_price:.2f}) outside target range "  
                            f"(${self.parameters['min_stock_price']:.2f} - ${self.parameters['max_stock_price']:.2f})")
                return signals
            
            # Exit signals for existing positions
            for position in self.current_positions:
                # Take profit when reached our target
                if position['unrealized_profit_pct'] >= self.parameters['profit_target_pct']:
                    signals['exit'] = True
                    signals['positions_to_close'].append(position['position_id'])
                    logger.info(f"Take profit signal for position {position['position_id']} "  
                               f"({position['unrealized_profit_pct']:.2%} gain)")
                
                # Close when approaching expiration
                elif position['days_to_expiration'] <= self.parameters['days_to_close']:
                    signals['exit'] = True
                    signals['positions_to_close'].append(position['position_id'])
                    logger.info(f"Closing position {position['position_id']} with {position['days_to_expiration']} days to expiry")
                
                # Roll when put is being tested (stock near strike) and close to expiration
                elif (position['days_to_expiration'] <= self.parameters['roll_days_threshold'] and
                      current_price <= position['strike'] * 1.02 and  # Within 2% of strike
                      self.parameters['roll_when_tested']):
                    # Mark for rolling rather than simple close
                    position['roll'] = True
                    signals['exit'] = True
                    signals['positions_to_close'].append(position['position_id'])
                    logger.info(f"Rolling position {position['position_id']} with {position['days_to_expiration']} days to expiry")
            
            # Entry signals - only if not at max positions
            if len(self.current_positions) >= self.parameters['max_positions']:
                logger.debug(f"At maximum positions ({self.parameters['max_positions']}), no new entries")
                return signals
            
            # Check market conditions for entry
            market_suitable = self._evaluate_market_conditions(data, indicators)
            
            if not market_suitable:
                logger.debug("Market conditions not suitable for new put sales")
                return signals
            
            # Get option chain
            option_chain = self._get_option_chain()
            if option_chain is None or option_chain.empty:
                logger.warning("Could not retrieve option chain")
                return signals
            
            # Select the best put contract
            contract = self.select_option_contract(data, option_chain)
            if contract is None:
                return signals
            
            # Set entry signal with the selected contract
            signals['entry'] = True
            signals['contract'] = contract
            
        except Exception as e:
            logger.error(f"Error generating Cash-Secured Put signals: {str(e)}")
        
        return signals
    
    def _evaluate_market_conditions(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> bool:
        """
        Evaluate if current market conditions are suitable for selling puts.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Boolean indicating if market conditions are suitable
        """
        try:
            # Check trend indicators
            trend_bullish = False
            if 'sma_20' in indicators and 'sma_50' in indicators:
                # Price above key moving averages suggests bullish trend
                price = data['close'].iloc[-1]
                trend_bullish = (price > indicators['sma_20'].iloc[-1] and 
                                price > indicators['sma_50'].iloc[-1])
            
            # Check volatility - high IV percentile is good for selling options
            iv_favorable = False
            if 'iv_percentile' in indicators:
                iv_favorable = indicators['iv_percentile'].iloc[-1] > 50
            
            # Check if we're not in extremely oversold conditions (avoid catching falling knives)
            not_oversold = True
            if 'rsi_14' in indicators:
                not_oversold = indicators['rsi_14'].iloc[-1] > 30
            
            # Check trend strength - moderate to strong trend is good
            trend_strength_good = False
            if 'trend_strength' in indicators:
                trend_strength = indicators['trend_strength'].iloc[-1]
                trend_strength_good = 40 <= trend_strength <= 80
            
            # Combine factors (can adjust weights as needed)
            # Must have at least 2 favorable conditions
            favorable_count = sum([trend_bullish, iv_favorable, not_oversold, trend_strength_good])
            return favorable_count >= 2
            
        except Exception as e:
            logger.error(f"Error evaluating market conditions: {str(e)}")
            return False
    
    def _get_option_chain(self) -> Optional[pd.DataFrame]:
        """
        Get the option chain for the current symbol.
        
        Returns:
            Option chain DataFrame or None if unavailable
        """
        try:
            if hasattr(self.session, 'get_option_chain'):
                return self.session.get_option_chain()
            else:
                logger.warning("Session doesn't support get_option_chain method")
                return None
        except Exception as e:
            logger.error(f"Error retrieving option chain: {str(e)}")
            return None
    
    def calculate_position_size(self, contract: Dict[str, Any]) -> int:
        """
        Calculate the appropriate position size (number of contracts) based on
        account constraints and risk parameters.
        
        Args:
            contract: Option contract details
            
        Returns:
            Number of contracts to trade
        """
        try:
            # Each contract is for 100 shares
            shares_per_contract = 100
            
            # Calculate capital required for one contract (cash to secure the put)
            capital_per_contract = contract['strike'] * shares_per_contract
            
            # Get account equity
            account_equity = self.get_account_equity()
            
            # Calculate maximum position size based on account percentage limit
            max_position_value = account_equity * self.parameters['max_position_size_pct']
            max_contracts_by_size = int(max_position_value / capital_per_contract)
            
            # Calculate maximum position size based on risk per trade
            # For puts, maximum risk is strike price - premium
            max_risk_amount = account_equity * self.parameters['risk_per_trade']
            max_risk_per_contract = (contract['strike'] - contract['mid']) * shares_per_contract
            max_contracts_by_risk = int(max_risk_amount / max_risk_per_contract) if max_risk_per_contract > 0 else 0
            
            # Apply account awareness constraints - ensure we have sufficient buying power
            buying_power = self.get_buying_power()
            max_contracts_by_buying_power = int(buying_power / capital_per_contract)
            
            # Use the most restrictive limit
            max_contracts = min(max_contracts_by_size, max_contracts_by_risk, max_contracts_by_buying_power)
            
            # Ensure at least 1 contract if possible, otherwise 0
            position_size = max(1, max_contracts) if max_contracts > 0 else 0
            
            # Check for sector exposure limits
            symbol = self.session.symbol
            sector = self._get_symbol_sector(symbol)
            if sector and position_size > 0:
                current_sector_exposure = self._get_sector_exposure(sector)
                max_sector_exposure = self.parameters['max_sector_exposure']
                
                # Calculate how many contracts would put us at the sector limit
                remaining_sector_capacity = max(0, max_sector_exposure - current_sector_exposure)
                sector_limit_contracts = int(remaining_sector_capacity * account_equity / capital_per_contract)
                
                position_size = min(position_size, sector_limit_contracts)
            
            logger.info(f"Calculated position size: {position_size} contracts "  
                       f"(${position_size * capital_per_contract:.2f})")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """
        Get the sector for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Sector name or empty string if not found
        """
        # This would use an actual data source or API in production
        # Simplified placeholder mapping for demonstration
        sector_map = {
            'AAPL': 'technology',
            'MSFT': 'technology',
            'GOOGL': 'technology',
            'AMZN': 'consumer_cyclical',
            'TSLA': 'automotive',
            'JPM': 'financial',
            'BAC': 'financial',
            'WMT': 'consumer_defensive',
            'PFE': 'healthcare',
            'XOM': 'energy'
        }
        return sector_map.get(symbol, "")
    
    def _get_sector_exposure(self, sector: str) -> float:
        """
        Get current exposure to a sector as percentage of account.
        
        Args:
            sector: Sector name
            
        Returns:
            Current exposure as decimal (0.0-1.0)
        """
        return self.sector_exposure.get(sector, 0.0)
    
    def _update_sector_exposure(self, sector: str, position_value: float) -> None:
        """
        Update the sector exposure after opening a new position.
        
        Args:
            sector: Sector name
            position_value: Dollar value of the position
        """
        current_exposure = self.sector_exposure.get(sector, 0.0)
        account_equity = self.get_account_equity()
        
        if account_equity > 0:
            new_exposure = current_exposure + (position_value / account_equity)
            self.sector_exposure[sector] = new_exposure
            logger.info(f"Updated {sector} exposure to {new_exposure:.2%}")
    
    def _execute_signals(self) -> None:
        """
        Execute the trading signals with account awareness checks.
        
        This method ensures we check for:
        1. Account balance requirements
        2. Cash to secure the puts
        3. Position size limits
        4. Sector exposure limits
        """
        # Ensure account status is up to date
        self.check_account_status()
        
        # Verify account has sufficient buying power
        buying_power = self.get_buying_power()
        if buying_power <= 0:
            logger.warning("Insufficient buying power for cash-secured put strategy")
            return
        
        # Execute exit signals first
        if self.signals.get('exit', False):
            for position_id in self.signals.get('positions_to_close', []):
                # Find the position details
                position = next((p for p in self.current_positions if p['position_id'] == position_id), None)
                
                if position:
                    if position.get('roll', False):
                        # Roll the position instead of just closing
                        self._roll_position(position)
                    else:
                        # Close the position normally
                        self._close_position(position_id)
                        logger.info(f"Closed position {position_id}")
                
        # Execute entry signals
        if self.signals.get('entry', False) and self.signals.get('contract'):
            contract = self.signals.get('contract')
            
            # Calculate position size
            position_size = self.calculate_position_size(contract)
            
            # Only proceed if position size > 0
            if position_size > 0:
                # Calculate total collateral required
                shares_per_contract = 100
                total_collateral = contract['strike'] * shares_per_contract * position_size
                
                # Final check of account balance for total collateral
                if total_collateral <= buying_power:
                    # Security type for cash-secured puts is always option
                    security_type = 'option'
                    
                    # Validate trade size
                    if self.validate_trade_size(contract['symbol'], position_size, contract['mid'], is_day_trade=False):
                        # Open the position
                        self._open_position(contract, position_size, total_collateral)
                    else:
                        logger.warning(f"Trade validation failed for {contract['symbol']}, size: {position_size}")
                else:
                    logger.warning(f"Insufficient buying power for collateral: ${total_collateral:.2f} required, ${buying_power:.2f} available")
    
    def _open_position(self, contract: Dict[str, Any], position_size: int, collateral: float) -> None:
        """
        Open a new cash-secured put position.
        
        Args:
            contract: Option contract details
            position_size: Number of contracts
            collateral: Total collateral required
        """
        try:
            # Generate a unique position ID
            position_id = f"CSP_{contract['symbol']}_{contract['strike']}_{contract['expiration']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Record entry price as the mid price (or ask in real implementation)
            entry_price = contract['mid']
            
            # Calculate max profit (premium received)
            max_profit = entry_price * 100 * position_size
            
            # Calculate max loss (strike - premium) * 100 * contracts
            max_loss = (contract['strike'] - entry_price) * 100 * position_size
            
            # Create position object
            position = {
                'position_id': position_id,
                'symbol': contract['symbol'],
                'strategy': 'cash_secured_put',
                'option_type': 'put',
                'strike': contract['strike'],
                'expiration': contract['expiration'],
                'days_to_expiration': contract['days_to_expiration'],
                'entry_date': datetime.now(),
                'entry_price': entry_price,
                'position_size': position_size,
                'collateral': collateral,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'unrealized_profit_pct': 0.0,
                'status': 'open'
            }
            
            # Add to current positions
            self.current_positions.append(position)
            
            # Update sector exposure
            symbol = contract['symbol']
            sector = self._get_symbol_sector(symbol)
            if sector:
                self._update_sector_exposure(sector, collateral)
            
            # In a real implementation, execute the trade through broker API
            if hasattr(self.session, 'sell_to_open_put'):
                self.session.sell_to_open_put(contract['symbol'], contract['strike'], 
                                            contract['expiration'], position_size)
            
            logger.info(f"Opened cash-secured put position: {position_size} contracts of {contract['symbol']} "
                       f"${contract['strike']} puts, expiring {contract['expiration']}, "
                       f"received ${max_profit:.2f} premium, collateral: ${collateral:.2f}")
            
        except Exception as e:
            logger.error(f"Error opening position: {str(e)}")
    
    def _close_position(self, position_id: str) -> None:
        """
        Close an existing cash-secured put position.
        
        Args:
            position_id: Unique identifier for the position
        """
        try:
            # Find the position in our list
            position = next((p for p in self.current_positions if p['position_id'] == position_id), None)
            
            if not position:
                logger.warning(f"Position {position_id} not found")
                return
            
            # Get current value of the option (would come from market data in real implementation)
            # Here we approximate based on days remaining and entry price
            days_passed = position['days_to_expiration'] - position.get('days_to_expiration', 0)
            time_decay_factor = days_passed / position['days_to_expiration'] if position['days_to_expiration'] > 0 else 0.5
            current_price = position['entry_price'] * (1 - time_decay_factor)  # Simplified time decay model
            
            # Calculate profit/loss
            profit = (position['entry_price'] - current_price) * 100 * position['position_size']
            profit_pct = profit / position['max_profit'] if position['max_profit'] > 0 else 0
            
            # Update position status
            position['status'] = 'closed'
            position['exit_date'] = datetime.now()
            position['exit_price'] = current_price
            position['profit'] = profit
            position['profit_pct'] = profit_pct
            
            # In a real implementation, execute the trade through broker API
            if hasattr(self.session, 'buy_to_close_put'):
                self.session.buy_to_close_put(position['symbol'], position['strike'], 
                                             position['expiration'], position['position_size'])
            
            # Update sector exposure
            sector = self._get_symbol_sector(position['symbol'])
            if sector:
                self._update_sector_exposure(sector, -position['collateral'])
            
            logger.info(f"Closed position {position_id} with ${profit:.2f} profit ({profit_pct:.2%})")
            
            # Remove from active positions list
            self.current_positions = [p for p in self.current_positions if p['position_id'] != position_id]
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
    
    def _roll_position(self, position: Dict[str, Any]) -> None:
        """
        Roll an existing cash-secured put position to a later expiration.
        
        Args:
            position: Position details
        """
        try:
            # First close the existing position
            self._close_position(position['position_id'])
            
            # Then get a new option chain
            option_chain = self._get_option_chain()
            if option_chain is None or option_chain.empty:
                logger.warning("Could not retrieve option chain for rolling position")
                return
            
            # Get current market data
            symbol = position['symbol']
            current_data = self.data_pipeline.get_data(symbol)
            if current_data.empty:
                logger.warning(f"Could not retrieve market data for {symbol}")
                return
            
            # Filter for new put options
            puts = option_chain[
                (option_chain['option_type'] == 'put') &
                # Days to expiration within our range, but later than current
                (option_chain['days_to_expiration'] > position['days_to_expiration']) &
                (option_chain['days_to_expiration'] <= self.parameters['days_to_expiration_max']) &
                # Similar delta (roll to similar delta)
                (option_chain['delta'].abs() >= self.parameters['roll_to_delta'] - 0.05) &
                (option_chain['delta'].abs() <= self.parameters['roll_to_delta'] + 0.05)
            ]
            
            if puts.empty:
                logger.warning("No suitable puts found for rolling")
                return
            
            # Sort by days to expiration (choose the one closest to our target DTE)
            puts = puts.sort_values('days_to_expiration')
            
            # Get the best candidate
            new_put = puts.iloc[0]
            
            # Create contract details for the new put
            new_contract = {
                'symbol': new_put['symbol'],
                'option_type': 'put',
                'strike': new_put['strike'],
                'expiration': new_put['expiration'],
                'days_to_expiration': new_put['days_to_expiration'],
                'delta': new_put['delta'],
                'bid': new_put['bid'],
                'ask': new_put['ask'],
                'mid': (new_put['bid'] + new_put['ask']) / 2,
                'open_interest': new_put['open_interest'],
                'volume': new_put['volume']
            }
            
            # Calculate position size (typically same as original)
            position_size = position['position_size']
            
            # Calculate collateral required
            shares_per_contract = 100
            collateral = new_contract['strike'] * shares_per_contract * position_size
            
            # Open the new position
            self._open_position(new_contract, position_size, collateral)
            
            logger.info(f"Rolled position from {position['strike']} strike, {position['expiration']} expiry " 
                       f"to {new_contract['strike']} strike, {new_contract['expiration']} expiry")
            
        except Exception as e:
            logger.error(f"Error rolling position: {str(e)}")
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible the cash-secured put strategy is with the current market regime.
        
        Args:
            market_regime: Current market regime description
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            'bullish': 0.85,              # Very good in bullish markets
            'neutral': 0.90,              # Excellent in neutral markets
            'bearish': 0.40,              # Poor in bearish markets
            'trending_up': 0.80,          # Good in uptrends
            'trending_down': 0.30,        # Poor in downtrends
            'ranging': 0.85,              # Very good in range-bound markets
            'volatile': 0.50,             # Moderate in volatile markets
            'low_volatility': 0.60,       # Above average in low volatility
            'high_volatility': 0.40,      # Poor in high volatility
            'high_iv': 0.95,              # Excellent in high IV environments
            'low_iv': 0.50,               # Moderate in low IV environments
            'oversold': 0.75,             # Good in oversold markets
            'overbought': 0.60,           # Above average in overbought markets
            'earnings_season': 0.65,      # Above average during earnings season
            'sector_rotation': 0.70,      # Good during sector rotations
        }
        
        # Default to moderate compatibility if regime not recognized
        return compatibility_map.get(market_regime, 0.60)
