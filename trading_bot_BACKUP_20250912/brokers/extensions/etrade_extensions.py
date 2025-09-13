"""
E*TRADE Broker Extensions

Implements broker-specific extensions for E*TRADE, providing access to
E*TRADE's unique features like portfolio analysis and option trading.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import the extension base classes
from trading_bot.brokers.broker_extensions import (
    AdvancedOptionsExtension,
    PortfolioAnalysisExtension
)

# Import event system components
from trading_bot.event_system import EventBus, Event

# Configure logging
logger = logging.getLogger(__name__)


class ETradeOptionsExtension(AdvancedOptionsExtension):
    """
    Implementation of advanced options trading capabilities for E*TRADE
    
    Provides access to option chains, expiration dates, strike prices, and
    multi-leg spread creation functionality.
    """
    
    def __init__(self, etrade_client=None):
        """
        Initialize the E*TRADE options extension
        
        Args:
            etrade_client: E*TRADE API client instance
        """
        self.client = etrade_client
        self.event_bus = EventBus()
        self._option_chain_cache = {}  # Symbol -> (timestamp, data) cache
        self._expiration_dates_cache = {}  # Symbol -> (timestamp, dates) cache
        self._cache_timeout = 300  # 5 minutes in seconds
        logger.info("E*TRADE Options Extension initialized")
    
    def get_option_chain(self, symbol: str, expiration_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get the full option chain for a symbol
        
        Args:
            symbol: Underlying symbol
            expiration_date: Specific expiration date (optional)
            
        Returns:
            Dict: Option chain data
        """
        cache_key = f"{symbol}_{expiration_date.strftime('%Y%m%d') if expiration_date else 'all'}"
        
        # Check cache first
        if cache_key in self._option_chain_cache:
            timestamp, data = self._option_chain_cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                logger.debug(f"Using cached option chain for {cache_key}")
                return data
        
        try:
            logger.info(f"Fetching option chain for {symbol} and expiration {expiration_date}")
            # Call E*TRADE API to get option chain data
            exp_date_str = expiration_date.strftime("%Y%m%d") if expiration_date else None
            option_chain = self.client.get_option_chains(
                symbol, 
                expiration=exp_date_str
            )
            
            # Cache the result
            self._option_chain_cache[cache_key] = (time.time(), option_chain)
            
            # Publish an event for the option chain retrieval
            self.event_bus.publish(Event(
                "option_chain_retrieved",
                {
                    "symbol": symbol,
                    "expiration_date": expiration_date,
                    "chain_size": len(option_chain.get("calls", [])) + len(option_chain.get("puts", []))
                }
            ))
            
            return option_chain
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {str(e)}")
            return {"error": str(e), "calls": [], "puts": []}
    
    def get_option_expiration_dates(self, symbol: str) -> List[datetime]:
        """
        Get available expiration dates for options on a symbol
        
        Args:
            symbol: Underlying symbol
            
        Returns:
            List[datetime]: Available expiration dates
        """
        # Check cache first
        if symbol in self._expiration_dates_cache:
            timestamp, dates = self._expiration_dates_cache[symbol]
            if time.time() - timestamp < self._cache_timeout:
                logger.debug(f"Using cached expiration dates for {symbol}")
                return dates
        
        try:
            logger.info(f"Fetching option expiration dates for {symbol}")
            # Call E*TRADE API to get expiration dates
            expiration_dates = self.client.get_option_expiration_dates(symbol)
            
            # Convert string dates to datetime objects if necessary
            parsed_dates = []
            for date_str in expiration_dates:
                try:
                    # Assuming date format from E*TRADE API is YYYYMMDD
                    parsed_date = datetime.strptime(date_str, "%Y%m%d")
                    parsed_dates.append(parsed_date)
                except ValueError:
                    logger.warning(f"Could not parse expiration date: {date_str}")
            
            # Cache the result
            self._expiration_dates_cache[symbol] = (time.time(), parsed_dates)
            
            return parsed_dates
            
        except Exception as e:
            logger.error(f"Error fetching option expiration dates for {symbol}: {str(e)}")
            return []
    
    def get_option_strikes(self, symbol: str, expiration_date: datetime) -> List[float]:
        """
        Get available strike prices for a symbol and expiration
        
        Args:
            symbol: Underlying symbol
            expiration_date: Option expiration date
            
        Returns:
            List[float]: Available strike prices
        """
        try:
            logger.info(f"Fetching option strikes for {symbol} and expiration {expiration_date}")
            
            # Get the option chain for this symbol and expiration
            option_chain = self.get_option_chain(symbol, expiration_date)
            
            # Extract unique strike prices from calls and puts
            strikes = set()
            for call in option_chain.get("calls", []):
                if "strikePrice" in call:
                    strikes.add(call["strikePrice"])
            
            for put in option_chain.get("puts", []):
                if "strikePrice" in put:
                    strikes.add(put["strikePrice"])
            
            # Sort strikes in ascending order
            sorted_strikes = sorted(list(strikes))
            return sorted_strikes
            
        except Exception as e:
            logger.error(f"Error fetching option strikes for {symbol} and {expiration_date}: {str(e)}")
            return []
    
    def create_option_spread(self, 
                           symbol: str,
                           spread_type: str,  # e.g., "vertical", "iron_condor", "butterfly"
                           expiration_date: datetime,
                           width: float,      # Distance between strikes
                           is_bullish: bool,  # Direction of the spread
                           quantity: int,
                           limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a multi-leg option spread order
        
        Args:
            symbol: Underlying symbol
            spread_type: Type of spread to create (e.g., "vertical", "iron_condor", "butterfly")
            expiration_date: Option expiration date
            width: Width between strikes
            is_bullish: Direction of the spread (True for bullish, False for bearish)
            quantity: Number of spreads to create
            limit_price: Optional limit price for the order
            
        Returns:
            Dict: Created order details
        """
        try:
            logger.info(f"Creating {spread_type} spread for {symbol} expiring {expiration_date}")
            
            # Get option chain data
            option_chain = self.get_option_chain(symbol, expiration_date)
            
            # Get underlying price
            underlying_price = self.client.get_quote(symbol)["last"]
            
            # Find appropriate strikes based on spread type and direction
            strikes = self.get_option_strikes(symbol, expiration_date)
            
            if not strikes:
                raise ValueError(f"No strikes available for {symbol} on {expiration_date}")
            
            # Find closest strike to current price
            closest_strike_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))
            closest_strike = strikes[closest_strike_idx]
            
            spread_legs = []
            
            # Create legs based on spread type
            if spread_type.lower() == "vertical":
                # Create a vertical spread (bull call or bear put)
                if is_bullish:
                    # Bull Call Spread (Buy lower strike call, sell higher strike call)
                    # Find appropriate strikes
                    if closest_strike_idx + 1 >= len(strikes):
                        raise ValueError("Cannot create bull call spread, no higher strike available")
                    
                    lower_strike = closest_strike
                    higher_strike = strikes[closest_strike_idx + 1]
                    
                    # Verify the width
                    if higher_strike - lower_strike != width:
                        # Find closest strike that matches the width
                        for i in range(closest_strike_idx + 1, len(strikes)):
                            if strikes[i] - lower_strike >= width:
                                higher_strike = strikes[i]
                                break
                    
                    # Create legs
                    spread_legs = [
                        {"option_type": "call", "strike": lower_strike, "action": "buy", "quantity": quantity},
                        {"option_type": "call", "strike": higher_strike, "action": "sell", "quantity": quantity}
                    ]
                else:
                    # Bear Put Spread (Buy higher strike put, sell lower strike put)
                    if closest_strike_idx - 1 < 0:
                        raise ValueError("Cannot create bear put spread, no lower strike available")
                    
                    higher_strike = closest_strike
                    lower_strike = strikes[closest_strike_idx - 1]
                    
                    # Verify the width
                    if higher_strike - lower_strike != width:
                        # Find closest strike that matches the width
                        for i in range(closest_strike_idx - 1, -1, -1):
                            if higher_strike - strikes[i] >= width:
                                lower_strike = strikes[i]
                                break
                    
                    # Create legs
                    spread_legs = [
                        {"option_type": "put", "strike": higher_strike, "action": "buy", "quantity": quantity},
                        {"option_type": "put", "strike": lower_strike, "action": "sell", "quantity": quantity}
                    ]
            
            elif spread_type.lower() == "iron_condor":
                # Iron Condor (Sell OTM Put Credit Spread + Sell OTM Call Credit Spread)
                
                # Find put spread strikes (below current price)
                put_strikes = [s for s in strikes if s < underlying_price]
                if len(put_strikes) < 2:
                    raise ValueError("Not enough strikes below current price for iron condor")
                
                # Find call spread strikes (above current price)
                call_strikes = [s for s in strikes if s > underlying_price]
                if len(call_strikes) < 2:
                    raise ValueError("Not enough strikes above current price for iron condor")
                
                # Select strikes based on width
                put_short_strike = put_strikes[-1]  # Closest to money
                put_long_strike = next((s for s in reversed(put_strikes[:-1]) if put_short_strike - s >= width), None)
                
                call_short_strike = call_strikes[0]  # Closest to money
                call_long_strike = next((s for s in call_strikes[1:] if s - call_short_strike >= width), None)
                
                if not (put_long_strike and call_long_strike):
                    raise ValueError(f"Cannot create iron condor with width {width}")
                
                # Create legs
                spread_legs = [
                    {"option_type": "put", "strike": put_long_strike, "action": "buy", "quantity": quantity},
                    {"option_type": "put", "strike": put_short_strike, "action": "sell", "quantity": quantity},
                    {"option_type": "call", "strike": call_short_strike, "action": "sell", "quantity": quantity},
                    {"option_type": "call", "strike": call_long_strike, "action": "buy", "quantity": quantity}
                ]
                
            elif spread_type.lower() == "butterfly":
                # Butterfly Spread (Buy Lower Strike, Sell 2x Middle Strike, Buy Higher Strike)
                
                # Need at least 3 strikes
                if len(strikes) < 3:
                    raise ValueError("Not enough strikes available for butterfly spread")
                
                middle_idx = closest_strike_idx
                
                # Ensure we have strikes on both sides
                if middle_idx == 0 or middle_idx == len(strikes) - 1:
                    middle_idx = len(strikes) // 2
                
                middle_strike = strikes[middle_idx]
                
                # Find wing strikes based on width
                lower_strike = None
                for i in range(middle_idx - 1, -1, -1):
                    if middle_strike - strikes[i] >= width:
                        lower_strike = strikes[i]
                        break
                
                higher_strike = None
                for i in range(middle_idx + 1, len(strikes)):
                    if strikes[i] - middle_strike >= width:
                        higher_strike = strikes[i]
                        break
                
                if not (lower_strike and higher_strike):
                    raise ValueError(f"Cannot create butterfly with width {width}")
                
                # Adjust for symmetry if possible
                if middle_strike - lower_strike != higher_strike - middle_strike:
                    # Try to find more symmetric strikes
                    lower_diff = middle_strike - lower_strike
                    higher_diff = higher_strike - middle_strike
                    
                    if lower_diff > higher_diff:
                        # Try to find a better lower strike
                        for i in range(middle_idx - 1, -1, -1):
                            if middle_strike - strikes[i] <= higher_diff * 1.2:  # Allow 20% tolerance
                                lower_strike = strikes[i]
                                break
                    else:
                        # Try to find a better higher strike
                        for i in range(middle_idx + 1, len(strikes)):
                            if strikes[i] - middle_strike <= lower_diff * 1.2:  # Allow 20% tolerance
                                higher_strike = strikes[i]
                                break
                
                # Create legs based on is_bullish (call butterfly or put butterfly)
                option_type = "call" if is_bullish else "put"
                spread_legs = [
                    {"option_type": option_type, "strike": lower_strike, "action": "buy", "quantity": quantity},
                    {"option_type": option_type, "strike": middle_strike, "action": "sell", "quantity": quantity * 2},
                    {"option_type": option_type, "strike": higher_strike, "action": "buy", "quantity": quantity}
                ]
            
            else:
                raise ValueError(f"Unsupported spread type: {spread_type}")
            
            # Create order with the calculated legs
            order_params = {
                "symbol": symbol,
                "order_type": "limit" if limit_price else "market",
                "limit_price": limit_price,
                "quantity": quantity,
                "spread_type": spread_type,
                "legs": spread_legs
            }
            
            # Place the order
            order_result = self.client.place_option_spread_order(order_params)
            
            # Publish an event for the spread creation
            self.event_bus.publish(Event(
                "option_spread_created",
                {
                    "symbol": symbol,
                    "spread_type": spread_type,
                    "expiration_date": expiration_date,
                    "is_bullish": is_bullish,
                    "quantity": quantity,
                    "order_id": order_result.get("order_id")
                }
            ))
            
            return order_result
            
        except Exception as e:
            logger.error(f"Error creating {spread_type} spread for {symbol}: {str(e)}")
            return {"error": str(e)}
    
    def get_extension_name(self) -> str:
        """Get the name of this extension"""
        return "ETradeOptionsExtension"
    
    def get_capabilities(self) -> Set[str]:
        """
        Get the set of capabilities provided by this extension
        
        Returns:
            Set[str]: Set of capability IDs
        """
        return {
            "option_chains", 
            "option_expirations", 
            "option_strikes", 
            "option_spreads",
            "vertical_spreads",
            "iron_condors",
            "butterfly_spreads"
        }


class ETradePortfolioExtension(PortfolioAnalysisExtension):
    """
    Implementation of portfolio analysis capabilities for E*TRADE
    
    Provides advanced portfolio risk metrics, position performance analysis,
    and correlation analysis.
    """
    
    def __init__(self, etrade_client=None):
        """
        Initialize the E*TRADE portfolio extension
        
        Args:
            etrade_client: E*TRADE API client instance
        """
        self.client = etrade_client
        self.event_bus = EventBus()
        self._risk_metrics_cache = None
        self._risk_metrics_timestamp = 0
        self._cache_timeout = 300  # 5 minutes in seconds
        logger.info("E*TRADE Portfolio Extension initialized")
    
    def get_portfolio_risk_metrics(self) -> Dict[str, float]:
        """
        Get risk metrics for the current portfolio
        
        Returns:
            Dict: Risk metrics (beta, VaR, etc.)
        """
        # Check cache first
        if self._risk_metrics_cache is not None:
            if time.time() - self._risk_metrics_timestamp < self._cache_timeout:
                logger.debug("Using cached portfolio risk metrics")
                return self._risk_metrics_cache
        
        try:
            logger.info("Calculating portfolio risk metrics")
            
            # Get portfolio positions
            positions = self.client.get_positions()
            
            # Calculate portfolio metrics
            portfolio_value = sum(position.get("marketValue", 0) for position in positions)
            
            # Calculate beta (weighted average of position betas)
            weighted_beta = 0
            for position in positions:
                symbol = position.get("symbol")
                market_value = position.get("marketValue", 0)
                
                # Get beta for this position
                beta = self.client.get_fundamental_data(symbol).get("beta", 1.0)
                
                # Add weighted beta
                if portfolio_value > 0:
                    weighted_beta += (market_value / portfolio_value) * beta
            
            # Calculate VaR (Value at Risk)
            # Simple historical VaR calculation (95% confidence, 1-day horizon)
            # Get historical returns for each position
            all_returns = []
            for position in positions:
                symbol = position.get("symbol")
                weight = position.get("marketValue", 0) / portfolio_value if portfolio_value > 0 else 0
                
                # Get 100 days of historical data
                historical_prices = self.client.get_historical_prices(symbol, days=100)
                
                # Calculate daily returns
                daily_returns = []
                for i in range(1, len(historical_prices)):
                    daily_return = (historical_prices[i] / historical_prices[i-1]) - 1
                    daily_returns.append(daily_return * weight)
                
                all_returns.append(daily_returns)
            
            # Combine weighted returns for portfolio
            portfolio_returns = [sum(day_returns) for day_returns in zip(*all_returns)]
            
            # Calculate 95% VaR
            var_95 = np.percentile(portfolio_returns, 5) * portfolio_value
            
            # Calculate other risk metrics
            volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
            sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
            
            # Compile metrics
            risk_metrics = {
                "portfolio_value": portfolio_value,
                "beta": weighted_beta,
                "var_95": abs(var_95),  # Make positive for easier interpretation
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "position_count": len(positions)
            }
            
            # Cache the result
            self._risk_metrics_cache = risk_metrics
            self._risk_metrics_timestamp = time.time()
            
            # Publish an event for the risk metrics calculation
            self.event_bus.publish(Event(
                "portfolio_risk_calculated",
                {
                    "portfolio_value": portfolio_value,
                    "beta": weighted_beta,
                    "var_95": abs(var_95),
                    "timestamp": datetime.now().isoformat()
                }
            ))
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {str(e)}")
            return {
                "error": str(e),
                "portfolio_value": 0,
                "beta": 1.0,
                "var_95": 0,
                "volatility": 0,
                "sharpe_ratio": 0,
                "position_count": 0
            }
    
    def get_position_performance(self, 
                               symbol: Optional[str] = None, 
                               timeframe: str = "1D",
                               start: Optional[datetime] = None,
                               end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get performance metrics for positions
        
        Args:
            symbol: Optional symbol to filter by
            timeframe: Analysis timeframe
            start: Start date
            end: End date
            
        Returns:
            DataFrame: Performance metrics
        """
        try:
            logger.info(f"Calculating position performance for timeframe {timeframe}")
            
            # Set default dates if not provided
            if end is None:
                end = datetime.now()
            if start is None:
                # Default to 30 days for daily timeframe
                if timeframe == "1D":
                    start = end - timedelta(days=30)
                # Default to 90 days for weekly timeframe
                elif timeframe == "1W":
                    start = end - timedelta(days=90)
                # Default to 365 days for monthly timeframe
                else:
                    start = end - timedelta(days=365)
            
            # Get positions
            if symbol:
                positions = [pos for pos in self.client.get_positions() if pos.get("symbol") == symbol]
            else:
                positions = self.client.get_positions()
            
            performance_data = []
            
            for position in positions:
                pos_symbol = position.get("symbol")
                quantity = position.get("quantity", 0)
                cost_basis = position.get("costBasis", 0)
                market_value = position.get("marketValue", 0)
                
                # Get historical data
                historical_data = self.client.get_historical_prices(
                    pos_symbol, 
                    start=start,
                    end=end,
                    interval=timeframe
                )
                
                # Calculate performance metrics
                if isinstance(historical_data, list) and len(historical_data) > 0:
                    # Calculate returns
                    total_return_pct = (market_value / cost_basis - 1) * 100 if cost_basis > 0 else 0
                    
                    # Calculate holding period in days
                    purchase_date = position.get("purchaseDate")
                    if purchase_date:
                        holding_period = (datetime.now() - purchase_date).days
                    else:
                        holding_period = 0
                    
                    # Calculate annualized return
                    if holding_period > 0:
                        annualized_return = ((1 + total_return_pct/100) ** (365/holding_period) - 1) * 100
                    else:
                        annualized_return = 0
                    
                    # Add to performance data
                    performance_data.append({
                        "symbol": pos_symbol,
                        "quantity": quantity,
                        "cost_basis": cost_basis,
                        "market_value": market_value,
                        "total_return_pct": total_return_pct,
                        "annualized_return": annualized_return,
                        "holding_period_days": holding_period
                    })
            
            # Convert to DataFrame
            performance_df = pd.DataFrame(performance_data)
            
            # Publish an event for the performance calculation
            self.event_bus.publish(Event(
                "position_performance_calculated",
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "position_count": len(performance_data),
                    "timestamp": datetime.now().isoformat()
                }
            ))
            
            return performance_df
            
        except Exception as e:
            logger.error(f"Error calculating position performance: {str(e)}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "symbol", "quantity", "cost_basis", "market_value",
                "total_return_pct", "annualized_return", "holding_period_days"
            ])
    
    def get_portfolio_correlation_matrix(self) -> pd.DataFrame:
        """
        Get correlation matrix for current portfolio holdings
        
        Returns:
            DataFrame: Correlation matrix
        """
        try:
            logger.info("Calculating portfolio correlation matrix")
            
            # Get positions
            positions = self.client.get_positions()
            symbols = [position.get("symbol") for position in positions]
            
            if not symbols:
                return pd.DataFrame()
            
            # Get historical data for all symbols (last 252 trading days)
            price_history = {}
            for symbol in symbols:
                price_history[symbol] = self.client.get_historical_prices(
                    symbol, 
                    days=252
                )
            
            # Create DataFrame of daily returns
            returns_data = {}
            for symbol, prices in price_history.items():
                # Calculate daily returns
                daily_returns = []
                for i in range(1, len(prices)):
                    daily_return = (prices[i] / prices[i-1]) - 1
                    daily_returns.append(daily_return)
                
                returns_data[symbol] = daily_returns
            
            # Create DataFrame and calculate correlation
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            # Publish an event for the correlation calculation
            self.event_bus.publish(Event(
                "portfolio_correlation_calculated",
                {
                    "symbol_count": len(symbols),
                    "avg_correlation": correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, 1)].mean(),
                    "timestamp": datetime.now().isoformat()
                }
            ))
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating portfolio correlation matrix: {str(e)}")
            return pd.DataFrame()
    
    def get_extension_name(self) -> str:
        """Get the name of this extension"""
        return "ETradePortfolioExtension"
    
    def get_capabilities(self) -> Set[str]:
        """
        Get the set of capabilities provided by this extension
        
        Returns:
            Set[str]: Set of capability IDs
        """
        return {
            "portfolio_risk", 
            "position_performance", 
            "portfolio_correlation"
        }
