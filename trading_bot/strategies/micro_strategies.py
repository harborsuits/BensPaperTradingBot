import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

from trading_bot.brokers.tradier_client import TradierClient

logger = logging.getLogger(__name__)

class MicroStrategy:
    """
    Base class for micro account trading strategies
    
    Designed for accounts under $1000 with focus on:
    - Strict risk management
    - Commission efficiency
    - Small position sizing
    """
    
    def __init__(self, 
                client: TradierClient,
                max_risk_per_trade: float = 0.01,  # 1% max risk per trade
                max_risk_total: float = 0.05,      # 5% max total risk
                min_rrr: float = 2.0,              # Minimum risk-reward ratio
                commission_per_trade: float = 1.0): # Typical commission per trade
        """
        Initialize the micro strategy
        
        Args:
            client: Tradier API client
            max_risk_per_trade: Maximum risk per trade as percentage of account
            max_risk_total: Maximum total risk percentage
            min_rrr: Minimum risk-reward ratio for trades
            commission_per_trade: Commission cost per trade
        """
        self.client = client
        self.max_risk_per_trade = max_risk_per_trade
        self.max_risk_total = max_risk_total
        self.min_rrr = min_rrr
        self.commission_per_trade = commission_per_trade
        
        # Strategy name
        self.name = "micro_base"
        
        # Market bias
        self.market_bias = "neutral"
        
        # Strategy state
        self.current_trades = []
        self.account_equity = 0
        
        # Update account equity
        self._update_account_info()
        
        logger.info(f"Initialized {self.name} strategy with max risk per trade: {max_risk_per_trade:.1%}")
    
    def _update_account_info(self):
        """Update account information"""
        try:
            account_balances = self.client.get_account_balances()
            self.account_equity = float(account_balances.get("equity", 0))
            
            # Get current positions
            self.current_positions = self.client.get_positions()
        except Exception as e:
            logger.error(f"Error updating account info: {str(e)}")
    
    def check_signal(self, symbol: str, data: Dict) -> Dict:
        """
        Check if there's a trading signal for the given symbol
        
        Args:
            symbol: Symbol to check
            data: Market data for analysis
            
        Returns:
            Signal dictionary (empty dict if no signal)
        """
        # Base class implementation returns no signal
        return {}
    
    def _is_commission_efficient(self, entry_price: float, stop_price: float, 
                              target_price: float, shares: int) -> bool:
        """
        Check if a trade is commission efficient
        
        Args:
            entry_price: Entry price
            stop_price: Stop loss price
            target_price: Target price
            shares: Number of shares
            
        Returns:
            True if the trade is commission efficient
        """
        # Calculate potential profit and loss
        max_loss = abs(entry_price - stop_price) * shares
        max_profit = abs(target_price - entry_price) * shares
        
        # Total commission (entry + exit)
        total_commission = self.commission_per_trade * 2
        
        # Check if potential profit covers commission costs with buffer
        commission_ratio = total_commission / max_profit if max_profit > 0 else float('inf')
        
        # Trade is commission efficient if the commission is less than 15% of potential profit
        return commission_ratio < 0.15
    
    def _position_size_from_risk(self, entry_price: float, stop_price: float) -> int:
        """
        Calculate position size based on risk
        
        Args:
            entry_price: Entry price
            stop_price: Stop loss price
            
        Returns:
            Number of shares to trade
        """
        # Risk amount in dollars
        risk_dollars = self.account_equity * self.max_risk_per_trade
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_price)
        
        # Calculate shares based on risk
        shares = int(risk_dollars / risk_per_share) if risk_per_share > 0 else 0
        
        # Ensure minimum of 1 share
        return max(1, shares)
    
    def adjust_for_market_bias(self, signal: Dict, market_bias: str) -> Dict:
        """
        Adjust signal based on market bias
        
        Args:
            signal: Original signal
            market_bias: Current market bias ("bullish", "bearish", "neutral", "volatile")
            
        Returns:
            Adjusted signal dictionary
        """
        if not signal:
            return signal
        
        # Store original signal values
        orig_side = signal.get("side")
        orig_shares = signal.get("shares", 0)
        orig_risk_pct = signal.get("risk_pct", self.max_risk_per_trade)
        
        # Default adjustments
        adjusted_signal = signal.copy()
        
        # Apply adjustments based on market bias
        if market_bias == "bullish":
            # In bullish market, take full size on long, reduce short positions
            if orig_side == "buy":
                # No change to long positions
                pass
            elif orig_side == "sell":
                # Reduce size of short positions by 50%
                adjusted_signal["shares"] = max(1, int(orig_shares * 0.5))
                adjusted_signal["risk_pct"] = orig_risk_pct * 0.5
        
        elif market_bias == "bearish":
            # In bearish market, take full size on short, reduce long positions
            if orig_side == "sell":
                # No change to short positions
                pass
            elif orig_side == "buy":
                # Reduce size of long positions by 50%
                adjusted_signal["shares"] = max(1, int(orig_shares * 0.5))
                adjusted_signal["risk_pct"] = orig_risk_pct * 0.5
        
        elif market_bias == "volatile":
            # In volatile market, reduce all position sizes by 30%
            adjusted_signal["shares"] = max(1, int(orig_shares * 0.7))
            adjusted_signal["risk_pct"] = orig_risk_pct * 0.7
            
        # Add market bias to signal metadata
        if "metadata" not in adjusted_signal:
            adjusted_signal["metadata"] = {}
        adjusted_signal["metadata"]["market_bias"] = market_bias
        
        # Log adjustment if any were made
        if adjusted_signal != signal:
            logger.info(f"Adjusted signal for {market_bias} market: {orig_shares} → {adjusted_signal.get('shares')} shares")
        
        return adjusted_signal


class MicroMomentumStrategy(MicroStrategy):
    """
    Momentum strategy for micro accounts
    
    Looks for strong trending moves with tight stop losses
    """
    
    def __init__(self, client: TradierClient, **kwargs):
        """Initialize the micro momentum strategy"""
        super().__init__(client, **kwargs)
        self.name = "micro_momentum"
        
        # Strategy parameters
        self.momentum_lookback = 5  # Number of days for momentum measurement
        self.volume_threshold = 1.5  # Volume must be this multiple of average
        self.ema_fast = 9            # Fast EMA period
        self.ema_slow = 20           # Slow EMA period
        
        logger.info(f"Initialized {self.name} strategy with {self.momentum_lookback} day lookback")
    
    def get_historical_data(self, symbol: str, lookback_days: int = 20) -> pd.DataFrame:
        """
        Get historical daily data for a symbol
        
        Args:
            symbol: Symbol to get data for
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Get historical data from Tradier
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_days + 10)).strftime('%Y-%m-%d')
            
            # Endpoint and parameters for historical data (not part of the provided TradierClient)
            # This would need to be implemented in the TradierClient class
            # For now we'll mock some data
            
            # Mock data for example purposes
            dates = pd.date_range(end=datetime.now(), periods=lookback_days)
            data = {
                'date': dates,
                'open': np.random.normal(100, 2, lookback_days),
                'high': np.random.normal(102, 2, lookback_days),
                'low': np.random.normal(98, 2, lookback_days),
                'close': np.random.normal(100, 2, lookback_days),
                'volume': np.random.normal(1000000, 200000, lookback_days)
            }
            
            df = pd.DataFrame(data)
            df = df.set_index('date')
            
            # Calculate indicators
            df['ema_fast'] = df['close'].ewm(span=self.ema_fast).mean()
            df['ema_slow'] = df['close'].ewm(span=self.ema_slow).mean()
            df['volume_ma'] = df['volume'].rolling(window=5).mean()
            df['momentum'] = df['close'].pct_change(periods=self.momentum_lookback)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def check_signal(self, symbol: str, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Check for momentum signals
        
        Args:
            symbol: Symbol to check
            data: Optional pre-loaded data DataFrame
            
        Returns:
            Signal dictionary with entry details (empty dict if no signal)
        """
        try:
            # Get historical data if not provided
            if data is None:
                data = self.get_historical_data(symbol)
            
            if data.empty:
                return {}
            
            # Get latest data
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            
            # Get current price
            current_price = latest['close']
            
            # Check for momentum entry conditions (long)
            long_signal = (
                latest['ema_fast'] > latest['ema_slow'] and      # Fast EMA above slow EMA
                latest['close'] > latest['ema_fast'] and         # Price above fast EMA
                latest['momentum'] > 0.03 and                    # Strong upward momentum (3%+)
                latest['volume'] > latest['volume_ma'] * self.volume_threshold  # High volume
            )
            
            # Check for momentum entry conditions (short)
            short_signal = (
                latest['ema_fast'] < latest['ema_slow'] and      # Fast EMA below slow EMA
                latest['close'] < latest['ema_fast'] and         # Price below fast EMA
                latest['momentum'] < -0.03 and                   # Strong downward momentum (-3%+)
                latest['volume'] > latest['volume_ma'] * self.volume_threshold  # High volume
            )
            
            # If we have a signal, prepare the details
            if long_signal or short_signal:
                # Determine side
                side = "buy" if long_signal else "sell"
                
                # Calculate stop loss and target
                # For long: Stop at recent low, target at 2x risk
                # For short: Stop at recent high, target at 2x risk
                if long_signal:
                    # Use recent low as stop, with some buffer
                    recent_low = min(data['low'].iloc[-3:])
                    stop_price = round(recent_low * 0.99, 2)  # 1% below recent low
                    risk_per_share = current_price - stop_price
                    target_price = round(current_price + (risk_per_share * self.min_rrr), 2)
                else:
                    # Use recent high as stop, with some buffer
                    recent_high = max(data['high'].iloc[-3:])
                    stop_price = round(recent_high * 1.01, 2)  # 1% above recent high
                    risk_per_share = stop_price - current_price
                    target_price = round(current_price - (risk_per_share * self.min_rrr), 2)
                
                # Calculate position size
                shares = self._position_size_from_risk(current_price, stop_price)
                
                # Check if trade is commission efficient
                is_efficient = self._is_commission_efficient(current_price, stop_price, target_price, shares)
                
                if not is_efficient:
                    logger.info(f"Signal for {symbol} rejected: Not commission efficient")
                    return {}
                
                # Create signal
                signal = {
                    "symbol": symbol,
                    "side": side,
                    "entry_price": current_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "shares": shares,
                    "risk_pct": self.max_risk_per_trade,
                    "strategy": self.name,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "momentum": latest['momentum'],
                        "volume_ratio": latest['volume'] / latest['volume_ma'],
                        "ema_difference": (latest['ema_fast'] / latest['ema_slow'] - 1) * 100  # Percentage difference
                    }
                }
                
                logger.info(f"Generated {side} signal for {symbol} at ${current_price:.2f}, "
                           f"stop: ${stop_price:.2f}, target: ${target_price:.2f}, shares: {shares}")
                
                return signal
            
            # No signal
            return {}
            
        except Exception as e:
            logger.error(f"Error checking momentum signal for {symbol}: {str(e)}")
            return {}


class MicroBreakoutStrategy(MicroStrategy):
    """
    Breakout strategy for micro accounts
    
    Looks for breakouts from consolidation patterns with tight stops
    """
    
    def __init__(self, client: TradierClient, **kwargs):
        """Initialize the micro breakout strategy"""
        super().__init__(client, **kwargs)
        self.name = "micro_breakout"
        
        # Strategy parameters
        self.consolidation_days = 5   # Number of days to check for consolidation
        self.breakout_threshold = 2.0  # Percentage threshold for breakout
        self.atr_multiple_stop = 1.0   # ATR multiple for stop loss
        self.min_volume_ratio = 1.5    # Minimum volume ratio for valid breakout
        
        logger.info(f"Initialized {self.name} strategy with {self.consolidation_days} day consolidation period")
    
    def get_historical_data(self, symbol: str, lookback_days: int = 20) -> pd.DataFrame:
        """
        Get historical daily data for a symbol
        
        Args:
            symbol: Symbol to get data for
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with historical data
        """
        try:
            # For example purposes, we'll use the same mock data function as above
            # In a real implementation, this would call the TradierClient
            
            # Mock data for example purposes
            dates = pd.date_range(end=datetime.now(), periods=lookback_days)
            data = {
                'date': dates,
                'open': np.random.normal(100, 2, lookback_days),
                'high': np.random.normal(102, 2, lookback_days),
                'low': np.random.normal(98, 2, lookback_days),
                'close': np.random.normal(100, 2, lookback_days),
                'volume': np.random.normal(1000000, 200000, lookback_days)
            }
            
            df = pd.DataFrame(data)
            df = df.set_index('date')
            
            # Calculate indicators
            df['atr'] = self._calculate_atr(df, 14)
            df['volume_ma'] = df['volume'].rolling(window=5).mean()
            df['high_5d'] = df['high'].rolling(window=self.consolidation_days).max()
            df['low_5d'] = df['low'].rolling(window=self.consolidation_days).min()
            df['range_5d'] = df['high_5d'] - df['low_5d']
            df['range_pct'] = df['range_5d'] / df['close']
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        
        Args:
            data: DataFrame with OHLC data
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def check_signal(self, symbol: str, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Check for breakout signals
        
        Args:
            symbol: Symbol to check
            data: Optional pre-loaded data DataFrame
            
        Returns:
            Signal dictionary with entry details (empty dict if no signal)
        """
        try:
            # Get historical data if not provided
            if data is None:
                data = self.get_historical_data(symbol)
            
            if data.empty:
                return {}
            
            # Get latest data
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            
            # Get current price
            current_price = latest['close']
            
            # Check if we're in a consolidation pattern (low volatility)
            is_consolidating = latest['range_pct'] < 0.05  # Less than 5% range over consolidation period
            
            # Check for breakout conditions
            upside_breakout = (
                is_consolidating and
                latest['close'] > prev['high_5d'] * (1 + self.breakout_threshold / 100) and  # Breakout above resistance
                latest['volume'] > latest['volume_ma'] * self.min_volume_ratio  # High volume
            )
            
            downside_breakout = (
                is_consolidating and
                latest['close'] < prev['low_5d'] * (1 - self.breakout_threshold / 100) and  # Breakout below support
                latest['volume'] > latest['volume_ma'] * self.min_volume_ratio  # High volume
            )
            
            # If we have a signal, prepare the details
            if upside_breakout or downside_breakout:
                # Determine side
                side = "buy" if upside_breakout else "sell"
                
                # Calculate stop loss using ATR
                atr = latest['atr']
                
                if upside_breakout:
                    # Stop loss is the consolidation low or ATR-based, whichever is tighter
                    atr_stop = current_price - (atr * self.atr_multiple_stop)
                    pattern_stop = prev['low_5d']
                    stop_price = max(atr_stop, pattern_stop)
                    risk_per_share = current_price - stop_price
                    target_price = current_price + (risk_per_share * self.min_rrr)
                else:
                    # Stop loss is the consolidation high or ATR-based, whichever is tighter
                    atr_stop = current_price + (atr * self.atr_multiple_stop)
                    pattern_stop = prev['high_5d']
                    stop_price = min(atr_stop, pattern_stop)
                    risk_per_share = stop_price - current_price
                    target_price = current_price - (risk_per_share * self.min_rrr)
                
                # Round prices to 2 decimal places
                stop_price = round(stop_price, 2)
                target_price = round(target_price, 2)
                
                # Calculate position size based on risk
                shares = self._position_size_from_risk(current_price, stop_price)
                
                # Check if trade is commission efficient
                is_efficient = self._is_commission_efficient(current_price, stop_price, target_price, shares)
                
                if not is_efficient:
                    logger.info(f"Signal for {symbol} rejected: Not commission efficient")
                    return {}
                
                # Create signal
                signal = {
                    "symbol": symbol,
                    "side": side,
                    "entry_price": current_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "shares": shares,
                    "risk_pct": self.max_risk_per_trade,
                    "strategy": self.name,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "breakout_type": "upside" if upside_breakout else "downside",
                        "consolidation_range_pct": latest['range_pct'] * 100,
                        "volume_ratio": latest['volume'] / latest['volume_ma'],
                        "atr": latest['atr']
                    }
                }
                
                logger.info(f"Generated {side} breakout signal for {symbol} at ${current_price:.2f}, "
                           f"stop: ${stop_price:.2f}, target: ${target_price:.2f}, shares: {shares}")
                
                return signal
            
            # No signal
            return {}
            
        except Exception as e:
            logger.error(f"Error checking breakout signal for {symbol}: {str(e)}")
            return {} 