import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

from trading_bot.strategies_new.stocks.base.stocks_base_strategy import StocksBaseStrategy
from trading_bot.strategies_new.validation.account_aware_mixin import AccountAwareMixin
from trading_bot.utils.math_utils import calculate_percentage_return

logger = logging.getLogger(__name__)

class ReversalStrategy(StocksBaseStrategy, AccountAwareMixin):
    """
    Reversal Strategy - Aims to identify and capitalize on potential trend reversals.
    
    The strategy focuses on:
    1. Identifying overbought/oversold conditions
    2. Spotting divergence between price and indicators
    3. Recognizing key reversal candlestick patterns
    4. Utilizing support/resistance levels for confirmation
    
    This strategy is account-aware, ensuring all trades comply with account status,
    regulatory constraints, and risk management principles.
    """
    
    def __init__(self, session=None, parameters=None, data_pipeline=None):
        """
        Initialize the Reversal Strategy with default parameters.
        
        Args:
            session: Trading session for executing orders
            parameters: Strategy parameters to override defaults
            data_pipeline: Data pipeline for retrieving market data
        """
        # Default parameters for the Reversal Strategy
        default_parameters = {
            # General settings
            'timeframe': '1h',            # Trading timeframe
            'max_positions': 5,           # Maximum number of concurrent positions
            
            # Position sizing and risk management
            'risk_per_trade': 0.01,       # Risk 1% of account per trade
            'stop_loss_pct': 0.02,        # 2% stop loss
            'trailing_stop_pct': 0.015,   # 1.5% trailing stop when in profit
            'take_profit_pct': 0.03,      # 3% take profit
            'max_position_size_pct': 0.05, # Maximum 5% of account per position
            
            # Oversold/Overbought conditions (for longs and shorts)
            'rsi_period': 14,
            'rsi_oversold': 30,           # RSI below this signals oversold
            'rsi_overbought': 70,         # RSI above this signals overbought
            
            # Moving averages for trend confirmation
            'fast_ma_period': 20,
            'slow_ma_period': 50,
            
            # Divergence detection
            'require_divergence': True,    # Require divergence for confirmation
            'divergence_lookback': 10,     # Bars to look back for divergence
            
            # Volume confirmation
            'require_volume_confirmation': True,
            'volume_threshold': 1.5,       # Require volume spike 1.5x average
            'volume_ma_period': 20,        # Period for volume moving average
            
            # Candlestick patterns for reversal confirmation
            'use_candlestick_patterns': True,
            
            # Advanced parameters
            'enable_short_selling': True,  # Whether to allow short positions
            'wait_for_confirmation': True, # Require additional confirmation candle
            
            # Volatility filter
            'max_atr_multiple': 2.0,       # Max ATR multiple for entry
            'atr_period': 14,              # ATR calculation period
        }
        
        # Call parent class initializers
        StocksBaseStrategy.__init__(self, session, 
                                   parameters if parameters else default_parameters, 
                                   data_pipeline)
        AccountAwareMixin.__init__(self)
        
        # Initialize state variables
        self.current_positions = []
        
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators required for the reversal strategy.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if data.empty:
            return indicators
        
        try:
            # Calculate RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=self.parameters['rsi_period']).mean()
            avg_loss = loss.rolling(window=self.parameters['rsi_period']).mean()
            
            rs = avg_gain / avg_loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate Moving Averages
            indicators['fast_ma'] = data['close'].rolling(window=self.parameters['fast_ma_period']).mean()
            indicators['slow_ma'] = data['close'].rolling(window=self.parameters['slow_ma_period']).mean()
            
            # Calculate ATR (Average True Range)
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift()).abs()
            low_close = (data['low'] - data['close'].shift()).abs()
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators['atr'] = tr.rolling(window=self.parameters['atr_period']).mean()
            
            # Volume Moving Average
            indicators['volume_ma'] = data['volume'].rolling(window=self.parameters['volume_ma_period']).mean()
            
            # Detect potential divergence
            indicators['bullish_divergence'] = self._detect_bullish_divergence(data, indicators['rsi'])
            indicators['bearish_divergence'] = self._detect_bearish_divergence(data, indicators['rsi'])
            
            # Detect potential candlestick patterns if enabled
            if self.parameters['use_candlestick_patterns']:
                indicators['bullish_reversal_pattern'] = self._detect_bullish_reversal_patterns(data)
                indicators['bearish_reversal_pattern'] = self._detect_bearish_reversal_patterns(data)
                
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            
        return indicators
    
    def _detect_bullish_divergence(self, data: pd.DataFrame, rsi: pd.Series) -> pd.Series:
        """
        Detect bullish divergence (price makes lower low but RSI makes higher low).
        
        Args:
            data: Market data
            rsi: RSI indicator values
            
        Returns:
            Series with True where bullish divergence is detected
        """
        lookback = self.parameters['divergence_lookback']
        result = pd.Series(False, index=data.index)
        
        if len(data) < lookback + 1:
            return result
        
        try:
            # Find local minima in price
            for i in range(lookback, len(data) - 1):
                # Check if we have a price lower low
                if (data['low'].iloc[i] < data['low'].iloc[i-lookback:i].min() and 
                    # But RSI shows higher low (bullish divergence)
                    rsi.iloc[i] > rsi.iloc[i-lookback:i].min() and
                    # And RSI is in oversold territory
                    rsi.iloc[i] < self.parameters['rsi_oversold']):
                    result.iloc[i] = True
                    
        except Exception as e:
            logger.error(f"Error detecting bullish divergence: {str(e)}")
            
        return result
    
    def _detect_bearish_divergence(self, data: pd.DataFrame, rsi: pd.Series) -> pd.Series:
        """
        Detect bearish divergence (price makes higher high but RSI makes lower high).
        
        Args:
            data: Market data
            rsi: RSI indicator values
            
        Returns:
            Series with True where bearish divergence is detected
        """
        lookback = self.parameters['divergence_lookback']
        result = pd.Series(False, index=data.index)
        
        if len(data) < lookback + 1:
            return result
        
        try:
            # Find local maxima in price
            for i in range(lookback, len(data) - 1):
                # Check if we have a price higher high
                if (data['high'].iloc[i] > data['high'].iloc[i-lookback:i].max() and 
                    # But RSI shows lower high (bearish divergence)
                    rsi.iloc[i] < rsi.iloc[i-lookback:i].max() and
                    # And RSI is in overbought territory
                    rsi.iloc[i] > self.parameters['rsi_overbought']):
                    result.iloc[i] = True
                    
        except Exception as e:
            logger.error(f"Error detecting bearish divergence: {str(e)}")
            
        return result
    
    def _detect_bullish_reversal_patterns(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect bullish reversal candlestick patterns.
        
        Args:
            data: Market data
            
        Returns:
            Series with True where bullish patterns are detected
        """
        result = pd.Series(False, index=data.index)
        
        if len(data) < 3:
            return result
        
        try:
            # Detect hammer pattern
            for i in range(2, len(data)):
                # Price is in a downtrend
                downtrend = (data['close'].iloc[i-2] > data['close'].iloc[i-1] > data['open'].iloc[i-1])
                
                # Current candle is a hammer
                body_size = abs(data['close'].iloc[i] - data['open'].iloc[i])
                lower_shadow = min(data['open'].iloc[i], data['close'].iloc[i]) - data['low'].iloc[i]
                upper_shadow = data['high'].iloc[i] - max(data['open'].iloc[i], data['close'].iloc[i])
                
                is_hammer = (
                    lower_shadow > 2 * body_size and  # Long lower shadow
                    upper_shadow < 0.5 * body_size and  # Small or no upper shadow
                    data['close'].iloc[i] > data['open'].iloc[i]  # Bullish close
                )
                
                # Bullish engulfing pattern
                is_bullish_engulfing = (
                    data['open'].iloc[i] < data['close'].iloc[i-1] and  # Open below previous close
                    data['close'].iloc[i] > data['open'].iloc[i-1] and  # Close above previous open
                    data['close'].iloc[i-1] < data['open'].iloc[i-1]  # Previous candle was bearish
                )
                
                # Morning star pattern (simplified)
                is_morning_star = (
                    data['close'].iloc[i-2] < data['open'].iloc[i-2] and  # First candle bearish
                    abs(data['close'].iloc[i-1] - data['open'].iloc[i-1]) < body_size * 0.5 and  # Second candle small
                    data['close'].iloc[i] > data['open'].iloc[i] and  # Third candle bullish
                    data['close'].iloc[i] > (data['open'].iloc[i-2] + data['close'].iloc[i-2]) / 2  # Closed above midpoint
                )
                
                if downtrend and (is_hammer or is_bullish_engulfing or is_morning_star):
                    result.iloc[i] = True
                    
        except Exception as e:
            logger.error(f"Error detecting bullish reversal patterns: {str(e)}")
            
        return result
    
    def _detect_bearish_reversal_patterns(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect bearish reversal candlestick patterns.
        
        Args:
            data: Market data
            
        Returns:
            Series with True where bearish patterns are detected
        """
        result = pd.Series(False, index=data.index)
        
        if len(data) < 3:
            return result
        
        try:
            # Detect shooting star pattern
            for i in range(2, len(data)):
                # Price is in an uptrend
                uptrend = (data['close'].iloc[i-2] < data['close'].iloc[i-1] < data['open'].iloc[i-1])
                
                # Current candle is a shooting star
                body_size = abs(data['close'].iloc[i] - data['open'].iloc[i])
                upper_shadow = data['high'].iloc[i] - max(data['open'].iloc[i], data['close'].iloc[i])
                lower_shadow = min(data['open'].iloc[i], data['close'].iloc[i]) - data['low'].iloc[i]
                
                is_shooting_star = (
                    upper_shadow > 2 * body_size and  # Long upper shadow
                    lower_shadow < 0.5 * body_size and  # Small or no lower shadow
                    data['close'].iloc[i] < data['open'].iloc[i]  # Bearish close
                )
                
                # Bearish engulfing pattern
                is_bearish_engulfing = (
                    data['open'].iloc[i] > data['close'].iloc[i-1] and  # Open above previous close
                    data['close'].iloc[i] < data['open'].iloc[i-1] and  # Close below previous open
                    data['close'].iloc[i-1] > data['open'].iloc[i-1]  # Previous candle was bullish
                )
                
                # Evening star pattern (simplified)
                is_evening_star = (
                    data['close'].iloc[i-2] > data['open'].iloc[i-2] and  # First candle bullish
                    abs(data['close'].iloc[i-1] - data['open'].iloc[i-1]) < body_size * 0.5 and  # Second candle small
                    data['close'].iloc[i] < data['open'].iloc[i] and  # Third candle bearish
                    data['close'].iloc[i] < (data['open'].iloc[i-2] + data['close'].iloc[i-2]) / 2  # Closed below midpoint
                )
                
                if uptrend and (is_shooting_star or is_bearish_engulfing or is_evening_star):
                    result.iloc[i] = True
                    
        except Exception as e:
            logger.error(f"Error detecting bearish reversal patterns: {str(e)}")
            
        return result
    
    def identify_key_levels(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Identify support and resistance levels for reversal confirmation.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary with support and resistance levels
        """
        levels = {
            'support': [],
            'resistance': []
        }
        
        if data.empty or len(data) < 30:  # Need sufficient data
            return levels
            
        try:
            # Find price swing highs and lows
            for i in range(5, len(data) - 5):
                # Check for swing high (potential resistance)
                if (data['high'].iloc[i] > data['high'].iloc[i-5:i].max() and 
                    data['high'].iloc[i] > data['high'].iloc[i+1:i+6].max()):
                    # Add resistance level
                    levels['resistance'].append(data['high'].iloc[i])
                
                # Check for swing low (potential support)
                if (data['low'].iloc[i] < data['low'].iloc[i-5:i].min() and 
                    data['low'].iloc[i] < data['low'].iloc[i+1:i+6].min()):
                    # Add support level
                    levels['support'].append(data['low'].iloc[i])
            
            # Cluster similar levels (combine levels that are very close)
            if levels['support']:
                levels['support'] = self._cluster_price_levels(levels['support'])
            
            if levels['resistance']:
                levels['resistance'] = self._cluster_price_levels(levels['resistance'])
                
        except Exception as e:
            logger.error(f"Error identifying key levels: {str(e)}")
        
        return levels
    
    def _cluster_price_levels(self, prices: List[float], threshold_pct: float = 0.01) -> List[float]:
        """
        Cluster similar price levels together.
        
        Args:
            prices: List of price levels
            threshold_pct: Percentage threshold for clustering
            
        Returns:
            List of clustered price levels
        """
        if not prices:
            return []
            
        prices = sorted(prices)
        clusters = []
        current_cluster = [prices[0]]
        
        for i in range(1, len(prices)):
            if (prices[i] - prices[i-1]) / prices[i-1] <= threshold_pct:
                # Add to current cluster
                current_cluster.append(prices[i])
            else:
                # Finish current cluster and start new one
                clusters.append(sum(current_cluster) / len(current_cluster))  # Average of cluster
                current_cluster = [prices[i]]
        
        # Add the last cluster
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        return clusters
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for the Reversal Strategy.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            'entry_long': False,
            'entry_short': False,
            'exit_long': False,
            'exit_short': False,
            'stop_loss': None,
            'take_profit': None
        }
        
        if data.empty or not indicators or len(data) < 20:  # Need sufficient data
            return signals
        
        try:
            current_price = data['close'].iloc[-1]
            current_index = data.index[-1]
            
            # Get key support and resistance levels
            key_levels = self.identify_key_levels(data)
            
            # Update signals for existing positions first (check stops, etc.)
            for position in self.current_positions:
                if position['direction'] == 'long':
                    # Check for exit signals for longs
                    if indicators.get('bearish_divergence', pd.Series()).iloc[-1]:
                        signals['exit_long'] = True
                        logger.info(f"Exit long signal: Bearish divergence detected at {current_price:.2f}")
                    
                    elif indicators.get('bearish_reversal_pattern', pd.Series()).iloc[-1]:
                        signals['exit_long'] = True
                        logger.info(f"Exit long signal: Bearish reversal pattern at {current_price:.2f}")
                
                elif position['direction'] == 'short':
                    # Check for exit signals for shorts
                    if indicators.get('bullish_divergence', pd.Series()).iloc[-1]:
                        signals['exit_short'] = True
                        logger.info(f"Exit short signal: Bullish divergence detected at {current_price:.2f}")
                    
                    elif indicators.get('bullish_reversal_pattern', pd.Series()).iloc[-1]:
                        signals['exit_short'] = True
                        logger.info(f"Exit short signal: Bullish reversal pattern at {current_price:.2f}")
            
            # Only check for new entry signals if we haven't maxed out positions
            if len(self.current_positions) >= self.parameters['max_positions']:
                logger.debug(f"Maximum positions ({self.parameters['max_positions']}) reached, no new entries")
                return signals
            
            # Check for long entry (bullish reversal)
            bullish_conditions = [
                indicators.get('rsi', pd.Series()).iloc[-1] < self.parameters['rsi_oversold'],  # Oversold
                indicators.get('bullish_divergence', pd.Series()).iloc[-1],  # Bullish divergence
                indicators.get('bullish_reversal_pattern', pd.Series()).iloc[-1]  # Bullish pattern
            ]
            
            # Volume confirmation for long entry
            if self.parameters['require_volume_confirmation']:
                volume_confirmation = (data['volume'].iloc[-1] > 
                                     indicators['volume_ma'].iloc[-1] * self.parameters['volume_threshold'])
                bullish_conditions.append(volume_confirmation)
            
            # Check for proximity to support levels
            near_support = False
            for level in key_levels.get('support', []):
                if abs(current_price - level) / current_price < 0.02:  # Within 2% of support
                    near_support = True
                    break
            bullish_conditions.append(near_support)
            
            # Generate long entry signal if enough conditions are met
            required_conditions = 3  # Require at least 3 bullish conditions
            if sum(bool(cond) for cond in bullish_conditions) >= required_conditions:
                # Calculate stop loss and take profit
                atr_value = indicators['atr'].iloc[-1]
                stop_loss = current_price - (atr_value * 2)  # 2 ATR stop
                take_profit = current_price + (atr_value * 3)  # 3 ATR target (1.5 risk:reward)
                
                signals['entry_long'] = True
                signals['stop_loss'] = stop_loss
                signals['take_profit'] = take_profit
                logger.info(f"Long entry signal at {current_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
            
            # Check for short entry (bearish reversal) if short selling is enabled
            if self.parameters['enable_short_selling']:
                bearish_conditions = [
                    indicators.get('rsi', pd.Series()).iloc[-1] > self.parameters['rsi_overbought'],  # Overbought
                    indicators.get('bearish_divergence', pd.Series()).iloc[-1],  # Bearish divergence
                    indicators.get('bearish_reversal_pattern', pd.Series()).iloc[-1]  # Bearish pattern
                ]
                
                # Volume confirmation for short entry
                if self.parameters['require_volume_confirmation']:
                    volume_confirmation = (data['volume'].iloc[-1] > 
                                         indicators['volume_ma'].iloc[-1] * self.parameters['volume_threshold'])
                    bearish_conditions.append(volume_confirmation)
                
                # Check for proximity to resistance levels
                near_resistance = False
                for level in key_levels.get('resistance', []):
                    if abs(current_price - level) / current_price < 0.02:  # Within 2% of resistance
                        near_resistance = True
                        break
                bearish_conditions.append(near_resistance)
                
                # Generate short entry signal if enough conditions are met
                if sum(bool(cond) for cond in bearish_conditions) >= required_conditions:
                    # Calculate stop loss and take profit
                    atr_value = indicators['atr'].iloc[-1]
                    stop_loss = current_price + (atr_value * 2)  # 2 ATR stop
                    take_profit = current_price - (atr_value * 3)  # 3 ATR target (1.5 risk:reward)
                    
                    signals['entry_short'] = True
                    signals['stop_loss'] = stop_loss
                    signals['take_profit'] = take_profit
                    logger.info(f"Short entry signal at {current_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
        
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
        
        return signals
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, is_long: bool) -> int:
        """
        Calculate the appropriate position size based on account risk parameters.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price level
            is_long: True if long position, False if short position
            
        Returns:
            Position size (number of shares)
        """
        try:
            # Use account-aware features to get current equity
            account_equity = self.get_account_equity()
            
            # Calculate risk amount in dollars (risk per trade * account equity)
            risk_amount = account_equity * self.parameters['risk_per_trade']
            
            # Calculate risk per share
            if is_long:
                risk_per_share = entry_price - stop_loss
            else:  # Short position
                risk_per_share = stop_loss - entry_price
            
            if risk_per_share <= 0 or risk_per_share > entry_price * 0.1:  # Sanity check, max 10% risk per share
                logger.warning(f"Invalid risk per share: {risk_per_share:.2f}, limiting to 2% of price")
                risk_per_share = entry_price * 0.02  # Limit to 2% of price
            
            # Calculate position size
            position_size = int(risk_amount / risk_per_share)  # Integer number of shares
            
            # Apply account percentage limit
            max_position_value = account_equity * self.parameters['max_position_size_pct']
            max_shares_by_size = int(max_position_value / entry_price)
            position_size = min(position_size, max_shares_by_size)
            
            # Apply account awareness constraints - ensure we have sufficient buying power
            buying_power = self.get_buying_power(is_long == False)  # Day trade check if short
            max_shares_by_buying_power = int(buying_power / entry_price)
            position_size = min(position_size, max_shares_by_buying_power)
            
            # Minimum position size of 1 share if all checks pass
            if position_size <= 0:
                logger.warning("Position sizing resulted in zero shares, risk parameters may be too strict")
                return 0
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0
    
    def _execute_signals(self) -> None:
        """
        Execute trading signals with account awareness checks.
        
        This method ensures we check for:
        1. Account status and regulatory compliance
        2. Available buying power
        3. Position sizing based on risk parameters
        4. PDT rule compliance for day trades
        """
        # Ensure account status is up to date
        self.check_account_status()
        
        # Get current price from data source
        data = self.data_pipeline.get_data()
        if data.empty:
            logger.warning("Cannot execute signals: No data available")
            return
            
        current_price = data['close'].iloc[-1]
        
        # First handle exits (for risk management priority)
        self._process_exits(current_price)
        
        # Then handle entries
        self._process_entries(current_price)
    
    def _process_exits(self, current_price: float) -> None:
        """
        Process exit signals and manage trailing stops.
        
        Args:
            current_price: Current market price
        """
        positions_to_close = []
        
        for position in self.current_positions:
            close_position = False
            
            # Check for explicit exit signals
            if (position['direction'] == 'long' and self.signals.get('exit_long', False)) or \
               (position['direction'] == 'short' and self.signals.get('exit_short', False)):
                close_position = True
                logger.info(f"Closing {position['direction']} position due to exit signal")
            
            # Check stop loss
            elif (position['direction'] == 'long' and current_price <= position['stop_loss']) or \
                 (position['direction'] == 'short' and current_price >= position['stop_loss']):
                close_position = True
                logger.info(f"Closing {position['direction']} position: Stop loss triggered at {current_price:.2f}")
            
            # Check take profit
            elif (position['direction'] == 'long' and current_price >= position['take_profit']) or \
                 (position['direction'] == 'short' and current_price <= position['take_profit']):
                close_position = True
                logger.info(f"Closing {position['direction']} position: Take profit triggered at {current_price:.2f}")
            
            # Update trailing stop if enabled
            elif position.get('trailing_stop_active', False):
                if position['direction'] == 'long':
                    # Update trailing stop for longs (if price moves higher)
                    new_stop = current_price * (1 - self.parameters['trailing_stop_pct'])
                    if new_stop > position['stop_loss']:
                        position['stop_loss'] = new_stop
                        logger.debug(f"Updated trailing stop to {new_stop:.2f} for {position['position_id']}")
                
                elif position['direction'] == 'short':
                    # Update trailing stop for shorts (if price moves lower)
                    new_stop = current_price * (1 + self.parameters['trailing_stop_pct'])
                    if new_stop < position['stop_loss']:
                        position['stop_loss'] = new_stop
                        logger.debug(f"Updated trailing stop to {new_stop:.2f} for {position['position_id']}")
            
            # Check if we need to activate trailing stop (when in profit)
            elif not position.get('trailing_stop_active', False):
                profit_threshold = 0.01  # 1% profit to activate trailing stop
                
                if position['direction'] == 'long' and current_price > position['entry_price'] * (1 + profit_threshold):
                    position['trailing_stop_active'] = True
                    # Initial trailing stop placement
                    position['stop_loss'] = max(position['stop_loss'], current_price * (1 - self.parameters['trailing_stop_pct']))
                    logger.info(f"Activated trailing stop at {position['stop_loss']:.2f} for {position['position_id']}")
                
                elif position['direction'] == 'short' and current_price < position['entry_price'] * (1 - profit_threshold):
                    position['trailing_stop_active'] = True
                    # Initial trailing stop placement
                    position['stop_loss'] = min(position['stop_loss'], current_price * (1 + self.parameters['trailing_stop_pct']))
                    logger.info(f"Activated trailing stop at {position['stop_loss']:.2f} for {position['position_id']}")
            
            if close_position:
                positions_to_close.append(position)
        
        # Close positions that met exit criteria
        for position in positions_to_close:
            self._close_position(position, current_price)
    
    def _process_entries(self, current_price: float) -> None:
        """
        Process entry signals with account awareness.
        
        Args:
            current_price: Current market price
        """
        # Check for long entry signal
        if self.signals.get('entry_long', False):
            # Check if allowed by account status, risk parameters, etc.
            stop_loss = self.signals.get('stop_loss')
            if stop_loss is None or stop_loss >= current_price:  # Sanity check
                logger.warning("Invalid stop loss for long entry")
                return
                
            # Calculate position size based on risk
            position_size = self.calculate_position_size(current_price, stop_loss, is_long=True)
            
            if position_size > 0:
                # Validate the trade size against account constraints
                if self.validate_trade_size(self.session.symbol, position_size, current_price):
                    # Execute the long entry
                    self._open_long_position(current_price, stop_loss, position_size)
                else:
                    logger.warning("Trade validation failed, cannot execute long entry")
        
        # Check for short entry signal
        elif self.signals.get('entry_short', False) and self.parameters['enable_short_selling']:
            # Check if allowed by account status, risk parameters, etc.
            stop_loss = self.signals.get('stop_loss')
            if stop_loss is None or stop_loss <= current_price:  # Sanity check
                logger.warning("Invalid stop loss for short entry")
                return
                
            # Calculate position size based on risk
            position_size = self.calculate_position_size(current_price, stop_loss, is_long=False)
            
            if position_size > 0:
                # Validate the trade size against account constraints
                if self.validate_trade_size(self.session.symbol, position_size, current_price):
                    # Execute the short entry
                    self._open_short_position(current_price, stop_loss, position_size)
                else:
                    logger.warning("Trade validation failed, cannot execute short entry")
    
    def _open_long_position(self, entry_price: float, stop_loss: float, position_size: int) -> None:
        """
        Open a new long position.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            position_size: Number of shares
        """
        try:
            # Generate unique position ID
            position_id = f"RVSL_LONG_{self.session.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Calculate take profit level
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * 1.5)  # 1.5:1 reward-to-risk ratio
            
            # Create position object
            position = {
                'position_id': position_id,
                'symbol': self.session.symbol,
                'direction': 'long',
                'entry_price': entry_price,
                'entry_time': datetime.now(),
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trailing_stop_active': False,
                'status': 'open',
                'strategy': 'reversal'
            }
            
            # Add to current positions
            self.current_positions.append(position)
            
            # Execute the trade through broker API
            if hasattr(self.session, 'buy_to_open'):
                self.session.buy_to_open(self.session.symbol, position_size, entry_price)
            
            total_cost = entry_price * position_size
            max_risk = (entry_price - stop_loss) * position_size
            
            logger.info(f"Opened long position: {position_size} shares of {self.session.symbol} at ${entry_price:.2f}, "
                       f"stop: ${stop_loss:.2f}, target: ${take_profit:.2f}, risk: ${max_risk:.2f}")
            
        except Exception as e:
            logger.error(f"Error opening long position: {str(e)}")
    
    def _open_short_position(self, entry_price: float, stop_loss: float, position_size: int) -> None:
        """
        Open a new short position.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            position_size: Number of shares
        """
        try:
            # Generate unique position ID
            position_id = f"RVSL_SHORT_{self.session.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Calculate take profit level
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * 1.5)  # 1.5:1 reward-to-risk ratio
            
            # Create position object
            position = {
                'position_id': position_id,
                'symbol': self.session.symbol,
                'direction': 'short',
                'entry_price': entry_price,
                'entry_time': datetime.now(),
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trailing_stop_active': False,
                'status': 'open',
                'strategy': 'reversal'
            }
            
            # Add to current positions
            self.current_positions.append(position)
            
            # Execute the trade through broker API
            if hasattr(self.session, 'sell_to_open'):
                self.session.sell_to_open(self.session.symbol, position_size, entry_price)
            
            total_exposure = entry_price * position_size
            max_risk = (stop_loss - entry_price) * position_size
            
            logger.info(f"Opened short position: {position_size} shares of {self.session.symbol} at ${entry_price:.2f}, "
                       f"stop: ${stop_loss:.2f}, target: ${take_profit:.2f}, risk: ${max_risk:.2f}")
            
        except Exception as e:
            logger.error(f"Error opening short position: {str(e)}")
    
    def _close_position(self, position: Dict[str, Any], current_price: float) -> None:
        """
        Close an existing position.
        
        Args:
            position: Position details dictionary
            current_price: Current market price
        """
        try:
            position_id = position['position_id']
            # Mark position as closed
            position['status'] = 'closed'
            position['exit_price'] = current_price
            position['exit_time'] = datetime.now()
            
            # Calculate profit/loss
            if position['direction'] == 'long':
                profit = (current_price - position['entry_price']) * position['position_size']
                profit_pct = (current_price - position['entry_price']) / position['entry_price']
                
                # Execute the trade through broker API
                if hasattr(self.session, 'sell_to_close'):
                    self.session.sell_to_close(position['symbol'], position['position_size'], current_price)
                    
            else:  # short position
                profit = (position['entry_price'] - current_price) * position['position_size']
                profit_pct = (position['entry_price'] - current_price) / position['entry_price']
                
                # Execute the trade through broker API
                if hasattr(self.session, 'buy_to_close'):
                    self.session.buy_to_close(position['symbol'], position['position_size'], current_price)
            
            position['profit'] = profit
            position['profit_pct'] = profit_pct
            
            logger.info(f"Closed {position['direction']} position {position_id}: "
                       f"P/L ${profit:.2f} ({profit_pct:.2%})")
            
            # Remove from active positions
            self.current_positions = [p for p in self.current_positions if p['position_id'] != position_id]
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible this strategy is with the current market regime.
        
        Args:
            market_regime: Current market regime description
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            'trending_up': 0.35,          # Below average in strong uptrends
            'trending_down': 0.35,        # Below average in strong downtrends
            'ranging': 0.70,              # Good in range-bound markets
            'volatile': 0.80,             # Very good in volatile markets (lots of reversals)
            'low_volatility': 0.40,       # Below average in low volatility
            'high_volatility': 0.85,      # Excellent in high volatility
            'bullish': 0.40,              # Below average in strongly bullish markets
            'bearish': 0.40,              # Below average in strongly bearish markets
            'neutral': 0.65,              # Above average in neutral markets
            'oversold': 0.90,             # Excellent for bullish reversals in oversold conditions
            'overbought': 0.90,           # Excellent for bearish reversals in overbought conditions
        }
        
        # Default to moderate compatibility if regime not recognized
        return compatibility_map.get(market_regime, 0.50)
