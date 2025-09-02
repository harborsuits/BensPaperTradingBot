import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

from trading_bot.strategies_new.options.base.options_base_strategy import OptionsBaseStrategy
from trading_bot.strategies_new.validation.account_aware_mixin import AccountAwareMixin
from trading_bot.utils.math_utils import calculate_percentage_return

logger = logging.getLogger(__name__)

class MarriedPutStrategy(OptionsBaseStrategy, AccountAwareMixin):
    """
    Married Put Strategy - A protection strategy that combines long stock with a protective put.
    
    The strategy involves:
    1. Buying shares of a stock
    2. Buying a protective put option to limit downside risk
    
    This is essentially a form of portfolio insurance, allowing for unlimited upside potential
    while limiting the downside risk to the put strike minus the premium paid.
    """
    
    def __init__(self, session=None, parameters=None, data_pipeline=None):
        """
        Initialize the Married Put Strategy with default parameters.
        
        Args:
            session: Trading session for executing orders
            parameters: Strategy parameters to override defaults
            data_pipeline: Data pipeline for retrieving market data
        """
        # Default parameters for the Married Put Strategy
        default_parameters = {
            # Stock selection criteria
            'min_stock_price': 20.0,
            'max_stock_price': 500.0,
            'min_avg_volume': 500000,  # Minimum average daily volume
            
            # Put option parameters
            'put_strike_percent': 0.9,  # Put strike 10% below current price
            'days_to_expiration_min': 30,
            'days_to_expiration_max': 60,
            'min_open_interest': 50,
            'min_option_volume': 10,
            
            # Position sizing and risk management
            'max_position_size_pct': 0.05,  # Maximum 5% of account per position
            'max_positions': 10,  # Maximum number of concurrent positions
            'risk_per_trade': 0.01,  # Maximum risk per trade (1% of account)
            
            # Entry and exit conditions
            'profit_target_pct': 0.20,  # Exit when 20% profit reached
            'stop_loss_pct': 0.05,  # Exit when 5% loss from peak price
            'days_to_close': 5,  # Close position when 5 days until expiration
            
            # Sector exposure limits
            'max_sector_exposure': 0.25,  # Max 25% exposure to any sector
            
            # Technical criteria
            'require_uptrend': True,  # Only enter in uptrends
            'min_rsi': 40,  # Minimum RSI for entry
            'max_rsi': 70,  # Maximum RSI for entry
        }
        
        # Call parent class initializers
        OptionsBaseStrategy.__init__(self, session, 
                                   parameters if parameters else default_parameters, 
                                   data_pipeline)
        AccountAwareMixin.__init__(self)
        
        # Initialize state variables
        self.current_positions = []
        self.sector_exposure = {}  # Track sector exposure
        
    def calculate_iv_percentile(self, data: pd.DataFrame, lookback: int = 252) -> float:
        """
        Calculate the current IV percentile relative to historical IV.
        
        Args:
            data: DataFrame containing historical implied volatility
            lookback: Number of days to look back
            
        Returns:
            IV percentile (0-100)
        """
        if 'implied_volatility' not in data.columns or len(data) < lookback / 2:
            return 50.0  # Default to middle if insufficient data
        
        # Limit to lookback period
        iv_history = data['implied_volatility'].tail(lookback)
        
        # Current IV
        current_iv = iv_history.iloc[-1]
        
        # Historical range
        iv_min = iv_history.min()
        iv_max = iv_history.max()
        
        # Avoid division by zero
        if iv_max == iv_min:
            return 50.0
        
        # Calculate percentile
        iv_percentile = (current_iv - iv_min) / (iv_max - iv_min) * 100
        
        return iv_percentile
    
    def select_option_contract(self, data: pd.DataFrame, option_chain: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Select the appropriate put option for protection based on our parameters.
        
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
            
            # Calculate target strike price
            target_strike = current_price * self.parameters['put_strike_percent']
            
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
                # Minimum liquidity
                (puts['open_interest'] >= self.parameters['min_open_interest']) &
                (puts['volume'] >= self.parameters['min_option_volume'])
            ]
            
            if puts.empty:
                logger.warning("No suitable put options found matching criteria")
                return None
            
            # Find the put option with strike closest to our target
            puts['strike_distance'] = abs(puts['strike'] - target_strike)
            puts = puts.sort_values('strike_distance')
            
            # Get the best candidate
            best_put = puts.iloc[0]
            
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
                'volume': best_put['volume']
            }
            
            logger.info(f"Selected protective put: Strike=${contract['strike']:.2f}, "
                     f"Expiry={contract['days_to_expiration']} days, "
                     f"Delta={contract['delta']:.2f}, "
                     f"Price=${contract['mid']:.2f}")
            
            return contract
            
        except Exception as e:
            logger.error(f"Error selecting put option: {str(e)}")
            return None
        
    def _evaluate_market_conditions(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> bool:
        """
        Evaluate if current market conditions are suitable for entering a married put position.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Boolean indicating if market conditions are suitable
        """
        try:
            # Check if we require an uptrend
            if self.parameters['require_uptrend']:
                uptrend = False
                if 'sma_20' in indicators and 'sma_50' in indicators:
                    # Price above key moving averages suggests uptrend
                    price = data['close'].iloc[-1]
                    uptrend = (price > indicators['sma_20'].iloc[-1] and 
                               price > indicators['sma_50'].iloc[-1])
                    
                    # Also check if shorter MA is above longer MA
                    uptrend = uptrend and (indicators['sma_20'].iloc[-1] > indicators['sma_50'].iloc[-1])
                    
                if not uptrend:
                    logger.debug("Stock not in uptrend, skipping entry")
                    return False
            
            # Check RSI conditions
            if 'rsi_14' in indicators:
                rsi = indicators['rsi_14'].iloc[-1]
                if rsi < self.parameters['min_rsi'] or rsi > self.parameters['max_rsi']:
                    logger.debug(f"RSI {rsi:.2f} outside target range ({self.parameters['min_rsi']}-{self.parameters['max_rsi']})")
                    return False
            
            # Check for recent earnings or major events (simplified example)
            # In a real implementation, this would check for upcoming earnings, ex-div dates, etc.
            has_upcoming_event = False  # Placeholder for event check
            if has_upcoming_event:
                logger.debug("Upcoming earnings or major event, avoiding entry")
                return False
            
            # All conditions passed
            return True
            
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
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for the Married Put strategy.
        
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
            'stock_price': 0,
            'positions_to_close': []
        }
        
        if data.empty or not indicators:
            return signals
        
        try:
            current_price = data['close'].iloc[-1]
            signals['stock_price'] = current_price
            
            # Check if price meets our stock selection criteria
            if current_price < self.parameters['min_stock_price'] or current_price > self.parameters['max_stock_price']:
                logger.debug(f"Stock price (${current_price:.2f}) outside target range "  
                            f"(${self.parameters['min_stock_price']:.2f} - ${self.parameters['max_stock_price']:.2f})")
                return signals
            
            # Check if volume meets our criteria
            if 'volume' in data.columns:
                avg_volume = data['volume'].tail(20).mean()
                if avg_volume < self.parameters['min_avg_volume']:
                    logger.debug(f"Average volume ({avg_volume:.0f}) below minimum ({self.parameters['min_avg_volume']:.0f})")
                    return signals
            
            # Exit signals for existing positions
            for position in self.current_positions:
                # Check for profit target
                current_profit_pct = self._calculate_position_profit(position, current_price)
                
                # Take profit when reached our target
                if current_profit_pct >= self.parameters['profit_target_pct']:
                    signals['exit'] = True
                    signals['positions_to_close'].append(position['position_id'])
                    logger.info(f"Take profit signal for position {position['position_id']} "  
                               f"({current_profit_pct:.2%} gain)")
                
                # Check for stop loss (using trailing high)
                if 'highest_price' not in position:
                    position['highest_price'] = max(position['entry_price'], current_price)
                elif current_price > position['highest_price']:
                    position['highest_price'] = current_price
                
                # Calculate drawdown from highest point
                drawdown = (position['highest_price'] - current_price) / position['highest_price']
                if drawdown >= self.parameters['stop_loss_pct']:
                    signals['exit'] = True
                    signals['positions_to_close'].append(position['position_id'])
                    logger.info(f"Stop loss triggered for position {position['position_id']} "  
                               f"({drawdown:.2%} drawdown from ${position['highest_price']:.2f})")
                
                # Close when put is approaching expiration
                if position['days_to_expiration'] <= self.parameters['days_to_close']:
                    signals['exit'] = True
                    signals['positions_to_close'].append(position['position_id'])
                    logger.info(f"Closing position {position['position_id']} with {position['days_to_expiration']} days to expiry")
            
            # Entry signals - only if not at max positions
            if len(self.current_positions) >= self.parameters['max_positions']:
                logger.debug(f"At maximum positions ({self.parameters['max_positions']}), no new entries")
                return signals
            
            # Check market conditions for entry
            market_suitable = self._evaluate_market_conditions(data, indicators)
            
            if not market_suitable:
                logger.debug("Market conditions not suitable for new married put position")
                return signals
            
            # Get option chain
            option_chain = self._get_option_chain()
            if option_chain is None or option_chain.empty:
                logger.warning("Could not retrieve option chain")
                return signals
            
            # Select the best put contract for protection
            contract = self.select_option_contract(data, option_chain)
            if contract is None:
                return signals
            
            # Set entry signal with the selected contract
            signals['entry'] = True
            signals['contract'] = contract
            
        except Exception as e:
            logger.error(f"Error generating Married Put signals: {str(e)}")
        
        return signals
    
    def _calculate_position_profit(self, position: Dict[str, Any], current_price: float) -> float:
        """
        Calculate current profit percentage for a married put position.
        
        Args:
            position: Position details
            current_price: Current stock price
            
        Returns:
            Profit percentage as decimal
        """
        # Initial investment = stock purchase + put premium
        initial_investment = position['entry_price'] + position['put_cost']
        
        # Current value of stock
        stock_value = current_price
        
        # Current value of put (simplified calculation)
        # In a real implementation, this would come from market data
        put_value = max(0, position['put_strike'] - current_price)
        
        # Current total value
        current_value = stock_value + put_value
        
        # Calculate return
        return (current_value - initial_investment) / initial_investment
    
    def calculate_position_size(self, stock_price: float, put_price: float) -> int:
        """
        Calculate the appropriate position size (number of shares) based on
        account constraints and risk parameters.
        
        Args:
            stock_price: Current stock price
            put_price: Price of the protective put
            
        Returns:
            Number of shares to trade
        """
        try:
            # Total cost per share (stock + put)
            total_cost_per_share = stock_price + put_price
            
            # Get account equity
            account_equity = self.get_account_equity()
            
            # Calculate maximum position size based on account percentage limit
            max_position_value = account_equity * self.parameters['max_position_size_pct']
            max_shares_by_size = int(max_position_value / total_cost_per_share)
            
            # Calculate maximum position size based on risk per trade
            max_risk_amount = account_equity * self.parameters['risk_per_trade']
            # In a married put, the maximum risk is limited by the put strike
            # Risk per share = Entry price - Put strike + Put premium
            max_risk_per_share = (stock_price - (stock_price * self.parameters['put_strike_percent'])) + put_price
            max_shares_by_risk = int(max_risk_amount / max_risk_per_share) if max_risk_per_share > 0 else 0
            
            # Apply account awareness constraints - ensure we have sufficient buying power
            buying_power = self.get_buying_power(day_trade=False)
            max_shares_by_buying_power = int(buying_power / total_cost_per_share)
            
            # Use the most restrictive limit
            max_shares = min(max_shares_by_size, max_shares_by_risk, max_shares_by_buying_power)
            
            # Ensure minimum of 1 share if possible, otherwise 0
            position_size = max(1, max_shares) if max_shares > 0 else 0
            
            # Check for sector exposure limits
            symbol = self.session.symbol
            sector = self._get_symbol_sector(symbol)
            if sector and position_size > 0:
                current_sector_exposure = self._get_sector_exposure(sector)
                max_sector_exposure = self.parameters['max_sector_exposure']
                
                # Calculate how many shares would put us at the sector limit
                remaining_sector_capacity = max(0, max_sector_exposure - current_sector_exposure)
                sector_limit_shares = int(remaining_sector_capacity * account_equity / total_cost_per_share)
                
                position_size = min(position_size, sector_limit_shares)
            
            # Round to standard lot sizes (optional)
            if position_size >= 100:
                position_size = (position_size // 100) * 100
            
            logger.info(f"Calculated position size: {position_size} shares "  
                       f"(${position_size * total_cost_per_share:.2f})")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0
    
    def _execute_signals(self) -> None:
        """
        Execute the trading signals with account awareness checks.
        
        This method ensures we check for:
        1. Account balance requirements
        2. Available buying power for stock and option purchases
        3. Position size limits
        4. Sector exposure limits
        """
        # Ensure account status is up to date
        self.check_account_status()
        
        # Verify account has sufficient buying power
        buying_power = self.get_buying_power(day_trade=False)
        if buying_power <= 0:
            logger.warning("Insufficient buying power for married put strategy")
            return
        
        # Execute exit signals first
        if self.signals.get('exit', False):
            for position_id in self.signals.get('positions_to_close', []):
                # Close the position
                self._close_position(position_id)
                logger.info(f"Closed position {position_id}")
        
        # Execute entry signals
        if self.signals.get('entry', False) and self.signals.get('contract'):
            contract = self.signals.get('contract')
            stock_price = self.signals.get('stock_price', 0)
            
            # Skip if no valid stock price
            if stock_price <= 0:
                logger.warning("Invalid stock price for entry")
                return
            
            # Calculate position size
            put_price = contract['mid']
            position_size = self.calculate_position_size(stock_price, put_price)
            
            # Only proceed if position size > 0
            if position_size > 0:
                # Calculate total cost
                total_cost = (stock_price * position_size) + (put_price * position_size)
                
                # Final check of account balance
                if total_cost <= buying_power:
                    # Security type for stocks is 'equity'
                    security_type = 'equity'
                    
                    # Validate trade size
                    if self.validate_trade_size(self.session.symbol, position_size, stock_price, is_day_trade=False):
                        # Open the position
                        self._open_position(stock_price, contract, position_size)
                    else:
                        logger.warning(f"Trade validation failed for {self.session.symbol}, size: {position_size}")
                else:
                    logger.warning(f"Insufficient buying power: ${total_cost:.2f} required, ${buying_power:.2f} available")
    
    def _open_position(self, stock_price: float, contract: Dict[str, Any], position_size: int) -> None:
        """
        Open a new married put position.
        
        Args:
            stock_price: Current stock price
            contract: Put option contract details
            position_size: Number of shares to purchase
        """
        try:
            # Generate a unique position ID
            symbol = self.session.symbol
            position_id = f"MP_{symbol}_{contract['strike']}_{contract['expiration']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Put option price
            put_price = contract['mid']  # Using mid price for calculation
            
            # Calculate total costs
            stock_cost = stock_price * position_size
            put_cost = put_price * position_size
            total_cost = stock_cost + put_cost
            
            # Calculate maximum risk
            max_risk = ((stock_price - contract['strike']) + put_price) * position_size
            if max_risk < 0:  # In case put strike is above stock price
                max_risk = put_cost
            
            # Create position object
            position = {
                'position_id': position_id,
                'symbol': symbol,
                'strategy': 'married_put',
                'entry_date': datetime.now(),
                'entry_price': stock_price,
                'position_size': position_size,
                'put_strike': contract['strike'],
                'put_cost': put_price,  # Per share
                'expiration': contract['expiration'],
                'days_to_expiration': contract['days_to_expiration'],
                'stock_cost': stock_cost,
                'option_cost': put_cost,
                'total_cost': total_cost,
                'max_risk': max_risk,
                'status': 'open'
            }
            
            # Add to current positions
            self.current_positions.append(position)
            
            # Update sector exposure
            sector = self._get_symbol_sector(symbol)
            if sector:
                self._update_sector_exposure(sector, total_cost)
            
            # In a real implementation, execute the trades through broker API
            # 1. Buy the stock
            if hasattr(self.session, 'buy_to_open_equity'):
                self.session.buy_to_open_equity(symbol, position_size, stock_price)
            
            # 2. Buy the put option
            if hasattr(self.session, 'buy_to_open_put'):
                self.session.buy_to_open_put(symbol, contract['strike'], 
                                           contract['expiration'], position_size)
            
            logger.info(f"Opened married put position: {position_size} shares of {symbol} at ${stock_price:.2f}, "
                      f"protected with ${contract['strike']:.2f} puts, expiring {contract['expiration']}, "
                      f"total cost: ${total_cost:.2f}, max risk: ${max_risk:.2f}")
            
        except Exception as e:
            logger.error(f"Error opening position: {str(e)}")
    
    def _close_position(self, position_id: str) -> None:
        """
        Close an existing married put position.
        
        Args:
            position_id: Unique identifier for the position
        """
        try:
            # Find the position in our list
            position = next((p for p in self.current_positions if p['position_id'] == position_id), None)
            
            if not position:
                logger.warning(f"Position {position_id} not found")
                return
            
            # Get current stock price (would come from market data in real implementation)
            symbol = position['symbol']
            current_data = self.data_pipeline.get_data(symbol)
            if current_data.empty:
                logger.warning(f"Could not retrieve current price for {symbol}")
                return
                
            current_price = current_data['close'].iloc[-1]
            
            # Get current value of the put option (simplified calculation)
            # In a real implementation, this would come from live option data
            put_value = max(0, position['put_strike'] - current_price)
            if put_value < 0.05 and current_price > position['put_strike']:
                put_value = 0.05  # Minimum time value for OTM puts
            
            # Calculate profit/loss
            initial_value = position['total_cost']
            current_value = (current_price * position['position_size']) + (put_value * position['position_size'])
            profit = current_value - initial_value
            profit_pct = profit / initial_value
            
            # Update position status
            position['status'] = 'closed'
            position['exit_date'] = datetime.now()
            position['exit_price'] = current_price
            position['put_exit_value'] = put_value
            position['profit'] = profit
            position['profit_pct'] = profit_pct
            
            # In a real implementation, execute the trades through broker API
            # 1. Sell the stock
            if hasattr(self.session, 'sell_to_close_equity'):
                self.session.sell_to_close_equity(symbol, position['position_size'], current_price)
            
            # 2. Sell the put option
            if hasattr(self.session, 'sell_to_close_put'):
                self.session.sell_to_close_put(symbol, position['put_strike'], 
                                             position['expiration'], position['position_size'])
            
            # Update sector exposure
            sector = self._get_symbol_sector(position['symbol'])
            if sector:
                self._update_sector_exposure(sector, -position['total_cost'])
            
            logger.info(f"Closed position {position_id} with ${profit:.2f} profit ({profit_pct:.2%})")
            
            # Remove from active positions list
            self.current_positions = [p for p in self.current_positions if p['position_id'] != position_id]
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible the married put strategy is with the current market regime.
        
        Args:
            market_regime: Current market regime description
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            'bullish': 0.90,              # Excellent in bullish markets
            'neutral': 0.70,              # Good in neutral markets
            'bearish': 0.75,              # Good in bearish markets (due to protection)
            'trending_up': 0.85,          # Very good in uptrends
            'trending_down': 0.65,        # Above average in downtrends (due to protection)
            'ranging': 0.60,              # Above average in range-bound markets
            'volatile': 0.80,             # Very good in volatile markets (protection helps)
            'low_volatility': 0.50,       # Moderate in low volatility (protection may be overpriced)
            'high_volatility': 0.85,      # Very good in high volatility (protection valuable)
            'high_iv': 0.65,              # Above average but puts expensive in high IV
            'low_iv': 0.75,               # Good in low IV (puts cheaper)
            'oversold': 0.80,             # Very good in oversold markets
            'overbought': 0.60,           # Above average in overbought markets
            'earnings_season': 0.85,      # Very good during earnings season (protection for announcements)
            'sector_rotation': 0.75,      # Good during sector rotations
        }
        
        # Default to moderate compatibility if regime not recognized
        return compatibility_map.get(market_regime, 0.60)
