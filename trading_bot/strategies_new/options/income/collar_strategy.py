import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

from trading_bot.strategies_new.options.base.options_base_strategy import OptionsBaseStrategy
from trading_bot.strategies_new.validation.account_aware_mixin import AccountAwareMixin
from trading_bot.utils.math_utils import calculate_percentage_return

logger = logging.getLogger(__name__)

class CollarStrategy(OptionsBaseStrategy, AccountAwareMixin):
    """
    Collar Strategy - A protective strategy that combines long stock with both 
    a protective put and a covered call.
    
    The strategy involves:
    1. Owning shares of a stock
    2. Buying a protective put option to limit downside risk
    3. Selling a covered call option to generate income and cap upside
    
    This creates a "collar" around the position, limiting both potential losses 
    and gains, while potentially generating net income from the options.
    """
    
    def __init__(self, session=None, parameters=None, data_pipeline=None):
        """
        Initialize the Collar Strategy with default parameters.
        
        Args:
            session: Trading session for executing orders
            parameters: Strategy parameters to override defaults
            data_pipeline: Data pipeline for retrieving market data
        """
        # Default parameters for the Collar Strategy
        default_parameters = {
            # Stock selection criteria
            'min_stock_price': 20.0,
            'max_stock_price': 500.0,
            'min_avg_volume': 500000,  # Minimum average daily volume
            
            # Put option parameters (downside protection)
            'put_strike_percent': 0.9,  # Put strike 10% below current price
            'days_to_expiration_min': 30,
            'days_to_expiration_max': 60,
            'min_open_interest': 50,
            'min_option_volume': 10,
            
            # Call option parameters (upside cap & income)
            'call_strike_percent': 1.1,  # Call strike 10% above current price
            
            # Position sizing and risk management
            'max_position_size_pct': 0.05,  # Maximum 5% of account per position
            'max_positions': 10,  # Maximum number of concurrent positions
            'risk_per_trade': 0.01,  # Maximum risk per trade (1% of account)
            
            # Entry and exit conditions
            'profit_target_pct': 0.15,  # Exit when 15% profit reached
            'stop_loss_pct': 0.05,  # Exit when 5% loss from peak price
            'days_to_close': 5,  # Close position when 5 days until expiration
            
            # Sector exposure limits
            'max_sector_exposure': 0.25,  # Max 25% exposure to any sector
            
            # Technical criteria for entry
            'require_uptrend': True,  # Only enter in uptrends
            'min_rsi': 40,  # Minimum RSI for entry
            'max_rsi': 70,  # Maximum RSI for entry
            
            # Options pricing
            'require_net_credit': False,  # If true, require call premium > put premium
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
    
    def select_option_contracts(self, data: pd.DataFrame, option_chain: pd.DataFrame) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Select the appropriate put and call options for the collar strategy.
        
        Args:
            data: Market data DataFrame
            option_chain: Option chain DataFrame
            
        Returns:
            Tuple of (put_contract, call_contract) or (None, None) if suitable options not found
        """
        try:
            if data.empty or option_chain.empty:
                return None, None
                
            # Get the current underlying price
            current_price = data['close'].iloc[-1]
            
            # Calculate target strike prices
            put_target_strike = current_price * self.parameters['put_strike_percent']
            call_target_strike = current_price * self.parameters['call_strike_percent']
            
            # Filter options by type
            puts = option_chain[option_chain['option_type'] == 'put']
            calls = option_chain[option_chain['option_type'] == 'call']
            
            if puts.empty or calls.empty:
                logger.warning("Options chain missing put or call options")
                return None, None
                
            # Apply selection criteria to puts
            puts = puts[
                # Days to expiration within our range
                (puts['days_to_expiration'] >= self.parameters['days_to_expiration_min']) &
                (puts['days_to_expiration'] <= self.parameters['days_to_expiration_max']) &
                # Minimum liquidity
                (puts['open_interest'] >= self.parameters['min_open_interest']) &
                (puts['volume'] >= self.parameters['min_option_volume'])
            ]
            
            # Apply same criteria to calls
            calls = calls[
                # Days to expiration within our range - match the put expiration
                (calls['days_to_expiration'] >= self.parameters['days_to_expiration_min']) &
                (calls['days_to_expiration'] <= self.parameters['days_to_expiration_max']) &
                # Minimum liquidity
                (calls['open_interest'] >= self.parameters['min_open_interest']) &
                (calls['volume'] >= self.parameters['min_option_volume'])
            ]
            
            if puts.empty or calls.empty:
                logger.warning("No suitable options found meeting criteria")
                return None, None
            
            # Find the put option with strike closest to our target
            puts['strike_distance'] = abs(puts['strike'] - put_target_strike)
            puts = puts.sort_values('strike_distance')
            selected_put = puts.iloc[0]
            
            # For the call, try to match expiration date with the put
            put_expiration = selected_put['expiration']
            matching_calls = calls[calls['expiration'] == put_expiration]
            
            if matching_calls.empty:
                logger.warning("No matching call options found for selected put expiration")
                return None, None
            
            # Find the call with strike closest to our target
            matching_calls['strike_distance'] = abs(matching_calls['strike'] - call_target_strike)
            matching_calls = matching_calls.sort_values('strike_distance')
            selected_call = matching_calls.iloc[0]
            
            # Create contract details for put
            put_contract = {
                'symbol': selected_put['symbol'],
                'option_type': 'put',
                'strike': selected_put['strike'],
                'expiration': selected_put['expiration'],
                'days_to_expiration': selected_put['days_to_expiration'],
                'delta': selected_put['delta'],
                'bid': selected_put['bid'],
                'ask': selected_put['ask'],
                'mid': (selected_put['bid'] + selected_put['ask']) / 2,
                'open_interest': selected_put['open_interest'],
                'volume': selected_put['volume']
            }
            
            # Create contract details for call
            call_contract = {
                'symbol': selected_call['symbol'],
                'option_type': 'call',
                'strike': selected_call['strike'],
                'expiration': selected_call['expiration'],
                'days_to_expiration': selected_call['days_to_expiration'],
                'delta': selected_call['delta'],
                'bid': selected_call['bid'],
                'ask': selected_call['ask'],
                'mid': (selected_call['bid'] + selected_call['ask']) / 2,
                'open_interest': selected_call['open_interest'],
                'volume': selected_call['volume']
            }
            
            # Check if we require a net credit collar
            if self.parameters['require_net_credit']:
                net_credit = call_contract['mid'] - put_contract['mid']
                if net_credit <= 0:
                    logger.debug(f"Net debit collar (${net_credit:.2f}), but strategy requires net credit")
                    return None, None
            
            logger.info(f"Selected collar: Put at ${put_contract['strike']:.2f}, Call at ${call_contract['strike']:.2f}, "
                      f"Expiry={put_contract['days_to_expiration']} days")
            
            return put_contract, call_contract
            
        except Exception as e:
            logger.error(f"Error selecting option contracts: {str(e)}")
            return None, None
    
    def _evaluate_market_conditions(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> bool:
        """
        Evaluate if current market conditions are suitable for a collar position.
        
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
            
            # Check volatility - collars work well in moderately volatile markets
            iv_favorable = True
            if 'iv_percentile' in indicators:
                iv = indicators['iv_percentile'].iloc[-1]
                if iv < 30 or iv > 80:
                    # Extremely low or high IV may not be ideal
                    logger.debug(f"IV percentile ({iv:.2f}) outside ideal range for collar")
                    iv_favorable = False
            
            # Check for upcoming earnings or major events (simplified example)
            has_upcoming_event = False  # Placeholder for event check
            
            # Combine factors - a collar might be good before events for protection
            if has_upcoming_event and iv_favorable:
                logger.info("Entering collar before upcoming event for protection")
                return True
            
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
        Generate trading signals for the Collar strategy.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            'entry': False,
            'exit': False,
            'put_contract': None,
            'call_contract': None,
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
                
                # Check for stop loss (based on total position value)
                if current_profit_pct <= -self.parameters['stop_loss_pct']:
                    signals['exit'] = True
                    signals['positions_to_close'].append(position['position_id'])
                    logger.info(f"Stop loss triggered for position {position['position_id']} "  
                               f"({current_profit_pct:.2%} loss)")
                
                # Close when options are approaching expiration
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
                logger.debug("Market conditions not suitable for new collar position")
                return signals
            
            # Get option chain
            option_chain = self._get_option_chain()
            if option_chain is None or option_chain.empty:
                logger.warning("Could not retrieve option chain")
                return signals
            
            # Select the best contracts for the collar
            put_contract, call_contract = self.select_option_contracts(data, option_chain)
            if put_contract is None or call_contract is None:
                return signals
            
            # Set entry signal with the selected contracts
            signals['entry'] = True
            signals['put_contract'] = put_contract
            signals['call_contract'] = call_contract
            
        except Exception as e:
            logger.error(f"Error generating Collar signals: {str(e)}")
        
        return signals
    
    def _calculate_position_profit(self, position: Dict[str, Any], current_price: float) -> float:
        """
        Calculate current profit percentage for a collar position.
        
        Args:
            position: Position details
            current_price: Current stock price
            
        Returns:
            Profit percentage as decimal
        """
        # Initial investment = stock purchase + put premium - call premium
        initial_investment = position['entry_price'] + position['put_cost'] - position['call_premium']
        
        # Current value of stock
        stock_value = current_price
        
        # Current value of put (simplified calculation)
        put_value = max(0, position['put_strike'] - current_price)
        if put_value < 0.05 and current_price > position['put_strike']:
            put_value = 0.05  # Minimum time value for OTM puts
        
        # Current value of call (obligation)
        call_obligation = max(0, current_price - position['call_strike'])
        
        # Current total value (stock + put - call obligation)
        current_value = stock_value + put_value - call_obligation
        
        # Calculate return
        return (current_value - initial_investment) / initial_investment
    
    def calculate_position_size(self, stock_price: float, put_price: float, call_price: float) -> int:
        """
        Calculate the appropriate position size (number of shares) based on
        account constraints and risk parameters.
        
        Args:
            stock_price: Current stock price
            put_price: Price of the protective put
            call_price: Premium received from selling the call
            
        Returns:
            Number of shares to trade
        """
        try:
            # Net cost per share (stock + put - call)
            net_cost_per_share = stock_price + put_price - call_price
            
            # Get account equity
            account_equity = self.get_account_equity()
            
            # Calculate maximum position size based on account percentage limit
            max_position_value = account_equity * self.parameters['max_position_size_pct']
            max_shares_by_size = int(max_position_value / stock_price)  # Based on full stock price
            
            # Calculate maximum position size based on risk per trade
            max_risk_amount = account_equity * self.parameters['risk_per_trade']
            
            # For a collar, the maximum risk is limited to:
            # (Entry price - Put strike) + Put premium - Call premium
            max_risk_per_share = (stock_price - (stock_price * self.parameters['put_strike_percent'])) + put_price - call_price
            # Ensure risk is not negative (can happen with net credit collars)
            max_risk_per_share = max(0.01, max_risk_per_share)
            
            max_shares_by_risk = int(max_risk_amount / max_risk_per_share)
            
            # Apply account awareness constraints - ensure we have sufficient buying power
            buying_power = self.get_buying_power(day_trade=False)
            # For a collar, we need buying power for the stock and the put (call is sold)
            cost_per_share_for_buying_power = stock_price + put_price
            max_shares_by_buying_power = int(buying_power / cost_per_share_for_buying_power)
            
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
                sector_limit_shares = int(remaining_sector_capacity * account_equity / stock_price)
                
                position_size = min(position_size, sector_limit_shares)
            
            # Round to standard lot sizes (optional)
            if position_size >= 100:
                position_size = (position_size // 100) * 100
            
            logger.info(f"Calculated position size: {position_size} shares "  
                       f"(${position_size * net_cost_per_share:.2f} net cost, "  
                       f"${position_size * stock_price:.2f} total exposure)")
            
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
            logger.warning("Insufficient buying power for collar strategy")
            return
        
        # Execute exit signals first
        if self.signals.get('exit', False):
            for position_id in self.signals.get('positions_to_close', []):
                # Close the position
                self._close_position(position_id)
                logger.info(f"Closed position {position_id}")
        
        # Execute entry signals
        if self.signals.get('entry', False) and self.signals.get('put_contract') and self.signals.get('call_contract'):
            put_contract = self.signals.get('put_contract')
            call_contract = self.signals.get('call_contract')
            stock_price = self.signals.get('stock_price', 0)
            
            # Skip if no valid stock price
            if stock_price <= 0:
                logger.warning("Invalid stock price for entry")
                return
            
            # Calculate position size
            put_price = put_contract['mid']
            call_price = call_contract['mid']
            position_size = self.calculate_position_size(stock_price, put_price, call_price)
            
            # Only proceed if position size > 0
            if position_size > 0:
                # Calculate total cost (stock + put - call)
                stock_cost = stock_price * position_size
                put_cost = put_price * position_size
                call_premium = call_price * position_size
                total_cost = stock_cost + put_cost - call_premium
                
                # Final check of account balance
                # For margin calculations, we need the full cost of stock + put
                margin_requirement = stock_cost + put_cost
                if margin_requirement <= buying_power:
                    # Security type for stocks is 'equity'
                    security_type = 'equity'
                    
                    # Validate trade size
                    if self.validate_trade_size(self.session.symbol, position_size, stock_price, is_day_trade=False):
                        # Open the position
                        self._open_position(stock_price, put_contract, call_contract, position_size)
                    else:
                        logger.warning(f"Trade validation failed for {self.session.symbol}, size: {position_size}")
                else:
                    logger.warning(f"Insufficient buying power: ${margin_requirement:.2f} required, ${buying_power:.2f} available")
    
    def _open_position(self, stock_price: float, put_contract: Dict[str, Any], call_contract: Dict[str, Any], position_size: int) -> None:
        """
        Open a new collar position.
        
        Args:
            stock_price: Current stock price
            put_contract: Put option contract details
            call_contract: Call option contract details
            position_size: Number of shares to purchase
        """
        try:
            # Generate a unique position ID
            symbol = self.session.symbol
            position_id = f"COLLAR_{symbol}_{put_contract['strike']}_{call_contract['strike']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Option prices
            put_price = put_contract['mid']  # Using mid price for calculation
            call_price = call_contract['mid']  # Using mid price for calculation
            
            # Calculate total costs
            stock_cost = stock_price * position_size
            put_cost = put_price * position_size
            call_premium = call_price * position_size
            total_cost = stock_cost + put_cost - call_premium
            
            # Calculate maximum risk and maximum profit
            # Max risk: (Stock price - Put strike) + Put premium - Call premium
            max_risk = ((stock_price - put_contract['strike']) + put_price - call_price) * position_size
            if max_risk < 0:  # In case of net credit collar with put strike close to stock price
                max_risk = 0
                
            # Max profit: (Call strike - Stock price) - Put premium + Call premium
            max_profit = ((call_contract['strike'] - stock_price) - put_price + call_price) * position_size
            
            # Create position object
            position = {
                'position_id': position_id,
                'symbol': symbol,
                'strategy': 'collar',
                'entry_date': datetime.now(),
                'entry_price': stock_price,
                'position_size': position_size,
                'put_strike': put_contract['strike'],
                'put_cost': put_price,  # Per share
                'call_strike': call_contract['strike'],
                'call_premium': call_price,  # Per share
                'expiration': put_contract['expiration'],  # Both options have same expiration
                'days_to_expiration': put_contract['days_to_expiration'],
                'stock_cost': stock_cost,
                'put_cost_total': put_cost,
                'call_premium_total': call_premium,
                'total_cost': total_cost,
                'max_risk': max_risk,
                'max_profit': max_profit,
                'status': 'open'
            }
            
            # Add to current positions
            self.current_positions.append(position)
            
            # Update sector exposure
            sector = self._get_symbol_sector(symbol)
            if sector:
                self._update_sector_exposure(sector, stock_cost)  # Use stock cost as the exposure amount
            
            # In a real implementation, execute the trades through broker API
            # 1. Buy the stock
            if hasattr(self.session, 'buy_to_open_equity'):
                self.session.buy_to_open_equity(symbol, position_size, stock_price)
            
            # 2. Buy the put option
            if hasattr(self.session, 'buy_to_open_put'):
                self.session.buy_to_open_put(symbol, put_contract['strike'], 
                                           put_contract['expiration'], position_size)
            
            # 3. Sell the call option
            if hasattr(self.session, 'sell_to_open_call'):
                self.session.sell_to_open_call(symbol, call_contract['strike'], 
                                            call_contract['expiration'], position_size)
            
            logger.info(f"Opened collar position: {position_size} shares of {symbol} at ${stock_price:.2f}, "
                       f"protected with ${put_contract['strike']:.2f} puts, "
                       f"capped with ${call_contract['strike']:.2f} calls, "
                       f"expiring {put_contract['expiration']}, "
                       f"net cost: ${total_cost:.2f}, max risk: ${max_risk:.2f}, max profit: ${max_profit:.2f}")
            
        except Exception as e:
            logger.error(f"Error opening position: {str(e)}")
    
    def _close_position(self, position_id: str) -> None:
        """
        Close an existing collar position.
        
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
            
            # Get current value of options (simplified calculation)
            # In a real implementation, this would come from live option data
            put_value = max(0, position['put_strike'] - current_price)
            if put_value < 0.05 and current_price > position['put_strike']:
                put_value = 0.05  # Minimum time value for OTM puts
                
            call_obligation = max(0, current_price - position['call_strike'])
            
            # Calculate profit/loss
            initial_value = position['total_cost']
            current_value = (current_price * position['position_size']) + \
                           (put_value * position['position_size']) - \
                           (call_obligation * position['position_size'])
            profit = current_value - initial_value
            profit_pct = profit / abs(initial_value) if initial_value != 0 else 0
            
            # Update position status
            position['status'] = 'closed'
            position['exit_date'] = datetime.now()
            position['exit_price'] = current_price
            position['put_exit_value'] = put_value
            position['call_exit_obligation'] = call_obligation
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
            
            # 3. Buy back the call option
            if hasattr(self.session, 'buy_to_close_call'):
                self.session.buy_to_close_call(symbol, position['call_strike'], 
                                             position['expiration'], position['position_size'])
            
            # Update sector exposure
            sector = self._get_symbol_sector(position['symbol'])
            if sector:
                self._update_sector_exposure(sector, -position['stock_cost'])
            
            logger.info(f"Closed position {position_id} with ${profit:.2f} profit ({profit_pct:.2%})")
            
            # Remove from active positions list
            self.current_positions = [p for p in self.current_positions if p['position_id'] != position_id]
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible the collar strategy is with the current market regime.
        
        Args:
            market_regime: Current market regime description
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            'bullish': 0.70,              # Good in bullish markets but limited upside
            'neutral': 0.85,              # Very good in neutral markets
            'bearish': 0.75,              # Good in bearish markets (due to protection)
            'trending_up': 0.65,          # Above average in uptrends but capped
            'trending_down': 0.75,        # Good in downtrends (due to protection)
            'ranging': 0.90,              # Excellent in range-bound markets
            'volatile': 0.85,             # Very good in volatile markets (protection helps)
            'low_volatility': 0.55,       # Moderate in low volatility (protection may be overpriced)
            'high_volatility': 0.80,      # Very good in high volatility (protection valuable)
            'high_iv': 0.65,              # Above average but puts expensive in high IV
            'low_iv': 0.60,               # Above average in low IV
            'oversold': 0.70,             # Good in oversold markets
            'overbought': 0.65,           # Above average in overbought markets
            'earnings_season': 0.80,      # Very good during earnings season (protection for announcements)
            'sector_rotation': 0.70,      # Good during sector rotations
        }
        
        # Default to moderate compatibility if regime not recognized
        return compatibility_map.get(market_regime, 0.60)
