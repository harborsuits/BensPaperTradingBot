"""
Tradier Broker Extensions

Implements broker-specific extensions for Tradier, providing access to
Tradier's unique features like advanced options trading and market data.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import re

# Import the extension base classes
from trading_bot.brokers.broker_extensions import (
    AdvancedOptionsExtension,
    TechnicalIndicatorExtension
)

# Configure logging
logger = logging.getLogger(__name__)


class TradierOptionsExtension(AdvancedOptionsExtension):
    """
    Implements advanced options trading features for Tradier.
    
    Tradier provides comprehensive options chain data, expiration dates,
    and multi-leg options order capabilities.
    """
    
    def __init__(self, tradier_client):
        """
        Initialize the extension with a Tradier client instance
        
        Args:
            tradier_client: Instance of TradierClient
        """
        self.client = tradier_client
    
    def get_option_chain(self, symbol: str, expiration_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get the full option chain for a symbol
        
        Args:
            symbol: Underlying symbol
            expiration_date: Specific expiration date (optional)
            
        Returns:
            Dict: Option chain data
        """
        try:
            # Get expiration dates if none provided
            if not expiration_date:
                dates = self.get_option_expiration_dates(symbol)
                if not dates:
                    return {"symbol": symbol, "expirations": [], "options": []}
                expiration_date = dates[0]  # Use nearest expiration
            
            # Format date for Tradier API
            exp_date_str = expiration_date.strftime("%Y-%m-%d")
            
            # Make API request to get options chain
            params = {
                "symbol": symbol,
                "expiration": exp_date_str,
                "greeks": "true"  # Include Greeks in the response
            }
            
            response = self.client._make_request("GET", "/v1/markets/options/chains", params=params)
            
            # Parse response
            if 'options' not in response or 'option' not in response['options']:
                return {"symbol": symbol, "expiration": exp_date_str, "options": []}
            
            options_data = response['options']['option']
            
            # Separate calls and puts
            calls = []
            puts = []
            
            for option in options_data:
                option_type = option.get('option_type')
                
                # Parse option fields
                parsed_option = {
                    "symbol": option.get('symbol', ''),
                    "strike": float(option.get('strike', 0)),
                    "last_price": float(option.get('last', 0)),
                    "bid": float(option.get('bid', 0)),
                    "ask": float(option.get('ask', 0)),
                    "volume": int(option.get('volume', 0)),
                    "open_interest": int(option.get('open_interest', 0)),
                    "delta": float(option.get('greeks', {}).get('delta', 0)),
                    "gamma": float(option.get('greeks', {}).get('gamma', 0)),
                    "theta": float(option.get('greeks', {}).get('theta', 0)),
                    "vega": float(option.get('greeks', {}).get('vega', 0)),
                    "implied_volatility": float(option.get('greeks', {}).get('mid_iv', 0)) * 100,  # Convert to percentage
                }
                
                if option_type == 'call':
                    calls.append(parsed_option)
                elif option_type == 'put':
                    puts.append(parsed_option)
            
            # Sort by strike price
            calls = sorted(calls, key=lambda x: x['strike'])
            puts = sorted(puts, key=lambda x: x['strike'])
            
            # Build the final result
            result = {
                "symbol": symbol,
                "expiration": exp_date_str,
                "underlying_price": float(response.get('underlying', {}).get('last', 0)),
                "calls": calls,
                "puts": puts
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting option chain for {symbol}: {str(e)}")
            return {"symbol": symbol, "expiration": expiration_date.strftime("%Y-%m-%d") if expiration_date else None, "error": str(e)}
    
    def get_option_expiration_dates(self, symbol: str) -> List[datetime]:
        """
        Get available expiration dates for options on a symbol
        
        Args:
            symbol: Underlying symbol
            
        Returns:
            List[datetime]: Available expiration dates
        """
        try:
            # Make API request to get expiration dates
            params = {"symbol": symbol}
            response = self.client._make_request("GET", "/v1/markets/options/expirations", params=params)
            
            # Parse dates
            dates = []
            if 'expirations' in response and 'expiration' in response['expirations']:
                expiration_dates = response['expirations']['expiration']
                
                # Handle single expiration date (not in a list)
                if isinstance(expiration_dates, dict):
                    expiration_dates = [expiration_dates]
                
                for date_obj in expiration_dates:
                    date_str = date_obj.get('date', '')
                    if date_str:
                        try:
                            date = datetime.strptime(date_str, "%Y-%m-%d")
                            dates.append(date)
                        except ValueError:
                            logger.warning(f"Failed to parse expiration date: {date_str}")
            
            # Sort dates in ascending order (nearest first)
            dates = sorted(dates)
            
            return dates
            
        except Exception as e:
            logger.error(f"Error getting option expiration dates for {symbol}: {str(e)}")
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
            # Format date for Tradier API
            exp_date_str = expiration_date.strftime("%Y-%m-%d")
            
            # Make API request to get strike prices
            params = {
                "symbol": symbol,
                "expiration": exp_date_str
            }
            
            response = self.client._make_request("GET", "/v1/markets/options/strikes", params=params)
            
            # Parse strikes
            strikes = []
            if 'strikes' in response and 'strike' in response['strikes']:
                strike_prices = response['strikes']['strike']
                
                # Handle single strike (not in a list)
                if isinstance(strike_prices, (int, float, str)):
                    strike_prices = [strike_prices]
                
                for strike in strike_prices:
                    try:
                        strikes.append(float(strike))
                    except (ValueError, TypeError):
                        logger.warning(f"Failed to parse strike price: {strike}")
            
            # Sort strikes in ascending order
            strikes = sorted(strikes)
            
            return strikes
            
        except Exception as e:
            logger.error(f"Error getting option strikes for {symbol} on {expiration_date}: {str(e)}")
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
            spread_type: Type of spread to create
            expiration_date: Option expiration date
            width: Width between strikes
            is_bullish: True for bullish spread, False for bearish
            quantity: Number of spreads to trade
            limit_price: Optional limit price for the entire spread
            
        Returns:
            Dict: Order information
        """
        try:
            # Format date for option symbols
            exp_date_str = expiration_date.strftime("%y%m%d")  # Format as YYMMDD for option symbols
            
            # Get available strikes
            strikes = self.get_option_strikes(symbol, expiration_date)
            if not strikes or len(strikes) < 2:
                raise ValueError(f"Not enough strikes available for {symbol} on {expiration_date}")
            
            # Get underlying price
            # Make API request to get current quote
            quote_params = {"symbols": symbol}
            quote_response = self.client._make_request("GET", "/v1/markets/quotes", params=quote_params)
            
            if 'quotes' in quote_response and 'quote' in quote_response['quotes']:
                underlying_price = float(quote_response['quotes']['quote'].get('last', 0))
            else:
                raise ValueError(f"Failed to get current price for {symbol}")
            
            # Find the nearest strikes based on underlying price and spread width
            legs = []
            
            if spread_type.lower() == "vertical":
                # For vertical spread, need 2 strikes that are 'width' apart
                # Find the strike closest to ATM
                atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
                atm_index = strikes.index(atm_strike)
                
                # Determine number of strikes to move to get desired width
                strike_step = strikes[1] - strikes[0] if len(strikes) > 1 else width
                steps = max(1, round(width / strike_step))
                
                if is_bullish:  # Bull spread (buy lower strike, sell higher strike)
                    # For bull call spread
                    buy_strike = atm_strike
                    sell_strike = buy_strike + (steps * strike_step)
                    
                    # Check if the sell strike is available
                    if sell_strike not in strikes:
                        sell_strike = strikes[min(atm_index + steps, len(strikes) - 1)]
                    
                    buy_option = f"{symbol}{exp_date_str}C{int(buy_strike * 1000):08d}"
                    sell_option = f"{symbol}{exp_date_str}C{int(sell_strike * 1000):08d}"
                    
                    legs = [
                        {"option_symbol": buy_option, "side": "buy", "quantity": quantity},
                        {"option_symbol": sell_option, "side": "sell", "quantity": quantity}
                    ]
                else:  # Bear spread (buy higher strike, sell lower strike)
                    # For bear put spread
                    buy_strike = atm_strike
                    sell_strike = buy_strike - (steps * strike_step)
                    
                    # Check if the sell strike is available
                    if sell_strike not in strikes:
                        sell_strike = strikes[max(atm_index - steps, 0)]
                    
                    buy_option = f"{symbol}{exp_date_str}P{int(buy_strike * 1000):08d}"
                    sell_option = f"{symbol}{exp_date_str}P{int(sell_strike * 1000):08d}"
                    
                    legs = [
                        {"option_symbol": buy_option, "side": "buy", "quantity": quantity},
                        {"option_symbol": sell_option, "side": "sell", "quantity": quantity}
                    ]
            
            elif spread_type.lower() == "iron_condor":
                # For iron condor, need 4 strikes
                # Find the strike closest to ATM
                atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
                atm_index = strikes.index(atm_strike)
                
                # Determine number of strikes to move to get desired width
                strike_step = strikes[1] - strikes[0] if len(strikes) > 1 else width
                steps = max(1, round(width / strike_step))
                
                # Calculate iron condor strikes
                lower_short = max(atm_strike - (steps * strike_step), strikes[0])
                lower_long = max(lower_short - (steps * strike_step), strikes[0])
                upper_short = min(atm_strike + (steps * strike_step), strikes[-1])
                upper_long = min(upper_short + (steps * strike_step), strikes[-1])
                
                # Create legs
                lower_long_option = f"{symbol}{exp_date_str}P{int(lower_long * 1000):08d}"
                lower_short_option = f"{symbol}{exp_date_str}P{int(lower_short * 1000):08d}"
                upper_short_option = f"{symbol}{exp_date_str}C{int(upper_short * 1000):08d}"
                upper_long_option = f"{symbol}{exp_date_str}C{int(upper_long * 1000):08d}"
                
                legs = [
                    {"option_symbol": lower_long_option, "side": "buy", "quantity": quantity},
                    {"option_symbol": lower_short_option, "side": "sell", "quantity": quantity},
                    {"option_symbol": upper_short_option, "side": "sell", "quantity": quantity},
                    {"option_symbol": upper_long_option, "side": "buy", "quantity": quantity}
                ]
            
            elif spread_type.lower() == "butterfly":
                # For butterfly, need 3 strikes with equal distances
                # Find the strike closest to ATM
                atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
                atm_index = strikes.index(atm_strike)
                
                # Determine number of strikes to move to get desired width
                strike_step = strikes[1] - strikes[0] if len(strikes) > 1 else width
                steps = max(1, round(width / strike_step))
                
                # Calculate butterfly strikes
                if is_bullish:  # Use call options
                    lower_strike = atm_strike
                    middle_strike = min(atm_strike + (steps * strike_step), strikes[-1])
                    upper_strike = min(middle_strike + (steps * strike_step), strikes[-1])
                    
                    lower_option = f"{symbol}{exp_date_str}C{int(lower_strike * 1000):08d}"
                    middle_option = f"{symbol}{exp_date_str}C{int(middle_strike * 1000):08d}"
                    upper_option = f"{symbol}{exp_date_str}C{int(upper_strike * 1000):08d}"
                else:  # Use put options
                    lower_strike = max(atm_strike - (2 * steps * strike_step), strikes[0])
                    middle_strike = max(atm_strike - (steps * strike_step), strikes[0])
                    upper_strike = atm_strike
                    
                    lower_option = f"{symbol}{exp_date_str}P{int(lower_strike * 1000):08d}"
                    middle_option = f"{symbol}{exp_date_str}P{int(middle_strike * 1000):08d}"
                    upper_option = f"{symbol}{exp_date_str}P{int(upper_strike * 1000):08d}"
                
                legs = [
                    {"option_symbol": lower_option, "side": "buy", "quantity": quantity},
                    {"option_symbol": middle_option, "side": "sell", "quantity": quantity * 2},
                    {"option_symbol": upper_option, "side": "buy", "quantity": quantity}
                ]
            
            else:
                raise ValueError(f"Unsupported spread type: {spread_type}")
            
            # Create order
            order_params = {
                "class": "multileg",
                "symbol": symbol,
                "type": "limit" if limit_price else "market",
                "duration": "day",
                "option_symbol": [leg["option_symbol"] for leg in legs],
                "side": [leg["side"] for leg in legs],
                "quantity": [leg["quantity"] for leg in legs]
            }
            
            if limit_price:
                order_params["price"] = limit_price
            
            # Execute order preview
            preview_endpoint = "/v1/accounts/{account_id}/orders"
            preview_params = {"preview": "true", **order_params}
            
            preview_response = self.client._make_request("POST", preview_endpoint, data=preview_params)
            
            # Return preview information
            return {
                "spread_type": spread_type,
                "symbol": symbol,
                "expiration": expiration_date.strftime("%Y-%m-%d"),
                "legs": legs,
                "is_bullish": is_bullish,
                "underlying_price": underlying_price,
                "limit_price": limit_price,
                "quantity": quantity,
                "preview": preview_response
            }
            
        except Exception as e:
            logger.error(f"Error creating {spread_type} spread for {symbol}: {str(e)}")
            return {
                "error": str(e),
                "spread_type": spread_type,
                "symbol": symbol,
                "expiration": expiration_date.strftime("%Y-%m-%d") if expiration_date else None
            }
    
    def get_extension_name(self) -> str:
        return "TradierOptionsExtension"
    
    def get_capabilities(self) -> Set[str]:
        return {"option_chains", "option_spreads", "option_analytics"}


class TradierTechnicalIndicatorExtension(TechnicalIndicatorExtension):
    """
    Implements technical indicator analysis for Tradier.
    
    This extension calculates common technical indicators based on historical data.
    """
    
    def __init__(self, tradier_client):
        """
        Initialize the extension with a Tradier client instance
        
        Args:
            tradier_client: Instance of TradierClient
        """
        self.client = tradier_client
        self._available_indicators = {
            "SMA": self._calculate_sma,
            "EMA": self._calculate_ema,
            "RSI": self._calculate_rsi,
            "MACD": self._calculate_macd,
            "BOLLINGER": self._calculate_bollinger_bands,
            "ATR": self._calculate_atr,
            "STOCHASTIC": self._calculate_stochastic
        }
    
    def _calculate_sma(self, prices: pd.Series, params: Dict[str, Any]) -> pd.Series:
        """Calculate Simple Moving Average"""
        period = params.get("period", 20)
        return prices.rolling(window=period).mean()
    
    def _calculate_ema(self, prices: pd.Series, params: Dict[str, Any]) -> pd.Series:
        """Calculate Exponential Moving Average"""
        period = params.get("period", 20)
        return prices.ewm(span=period, adjust=False).mean()
    
    def _calculate_rsi(self, prices: pd.Series, params: Dict[str, Any]) -> pd.Series:
        """Calculate Relative Strength Index"""
        period = params.get("period", 14)
        # Calculate daily price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # First RSI value using SMA
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        fast_period = params.get("fast_period", 12)
        slow_period = params.get("slow_period", 26)
        signal_period = params.get("signal_period", 9)
        
        # Calculate EMAs
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Create DataFrame with all components
        result = pd.DataFrame({
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram
        })
        
        return result
    
    def _calculate_bollinger_bands(self, prices: pd.Series, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        period = params.get("period", 20)
        std_dev = params.get("std_dev", 2)
        
        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=period).mean()
        
        # Calculate standard deviation
        std = prices.rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        # Create DataFrame with all bands
        result = pd.DataFrame({
            "middle_band": middle_band,
            "upper_band": upper_band,
            "lower_band": lower_band
        })
        
        return result
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, params: Dict[str, Any]) -> pd.Series:
        """Calculate Average True Range"""
        period = params.get("period", 14)
        
        # Create price DataFrame
        df = pd.DataFrame({
            "high": high,
            "low": low,
            "close": close,
            "close_prev": close.shift(1)
        })
        
        # Calculate true range
        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = abs(df["high"] - df["close_prev"])
        df["tr3"] = abs(df["low"] - df["close_prev"])
        df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        
        # Calculate ATR
        atr = df["true_range"].rolling(window=period).mean()
        
        return atr
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        k_period = params.get("k_period", 14)
        d_period = params.get("d_period", 3)
        
        # Calculate %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D (SMA of %K)
        d = k.rolling(window=d_period).mean()
        
        # Create DataFrame with both lines
        result = pd.DataFrame({
            "k": k,
            "d": d
        })
        
        return result
    
    def get_technical_indicator(self, 
                              symbol: str,
                              indicator: str,
                              timeframe: str,
                              params: Dict[str, Any],
                              start: datetime,
                              end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get technical indicator values calculated by the broker
        
        Args:
            symbol: Symbol to calculate indicator for
            indicator: Indicator name
            timeframe: Bar timeframe for calculation
            params: Indicator-specific parameters
            start: Start date
            end: End date (default: now)
            
        Returns:
            DataFrame: Indicator values with timestamps
        """
        try:
            # Standardize indicator name (uppercase)
            indicator = indicator.upper()
            
            # Check if indicator is supported
            if indicator not in self._available_indicators:
                raise ValueError(f"Unsupported indicator: {indicator}")
            
            # Set default end date if not provided
            if not end:
                end = datetime.now()
            
            # Get historical data for calculation
            bars = self.client.get_bars(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end
            )
            
            if not bars:
                return pd.DataFrame()
            
            # Convert bars to DataFrame
            df = pd.DataFrame({
                "timestamp": [bar.timestamp for bar in bars],
                "open": [bar.open_price for bar in bars],
                "high": [bar.high_price for bar in bars],
                "low": [bar.low_price for bar in bars],
                "close": [bar.close_price for bar in bars],
                "volume": [bar.volume for bar in bars]
            })
            
            # Set timestamp as index
            df.set_index("timestamp", inplace=True)
            
            # Calculate the indicator using the appropriate function
            indicator_func = self._available_indicators[indicator]
            
            if indicator in ["SMA", "EMA", "RSI"]:
                result = indicator_func(df["close"], params)
                
                # Create DataFrame with indicator values
                if isinstance(result, pd.Series):
                    result_df = pd.DataFrame({indicator.lower(): result})
                else:
                    result_df = result
            
            elif indicator == "MACD":
                result_df = indicator_func(df["close"], params)
            
            elif indicator == "BOLLINGER":
                result_df = indicator_func(df["close"], params)
            
            elif indicator == "ATR":
                result_df = pd.DataFrame({
                    "atr": indicator_func(df["high"], df["low"], df["close"], params)
                })
            
            elif indicator == "STOCHASTIC":
                result_df = indicator_func(df["high"], df["low"], df["close"], params)
            
            # Reset index to include timestamp in the result
            result_df = result_df.reset_index()
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating {indicator} for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_available_indicators(self) -> List[str]:
        """
        Get list of available technical indicators from this broker
        
        Returns:
            List[str]: Available indicator names
        """
        return list(self._available_indicators.keys())
    
    def get_extension_name(self) -> str:
        return "TradierTechnicalIndicatorExtension"
    
    def get_capabilities(self) -> Set[str]:
        return {"technical_indicators"}
