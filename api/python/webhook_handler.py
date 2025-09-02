import os
import json
import logging
import hmac
import hashlib
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import threading
import pandas as pd

from trading_bot.brokers.tradier_client import TradierClient
from trading_bot.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class TradingViewSignal:
    """Data class for TradingView webhook signals"""
    def __init__(self, data: Dict[str, Any]):
        self.raw_data = data
        self.symbol = data.get('symbol')
        self.strategy = data.get('strategy')
        self.signal = data.get('signal')  # 'buy', 'sell', 'close', etc.
        self.timeframe = data.get('timeframe', '1d')
        self.price = data.get('price')
        self.timestamp = data.get('timestamp') or datetime.now().isoformat()
        self.stop_loss = data.get('stop_loss')
        self.take_profit = data.get('take_profit')
        self.risk_percent = data.get('risk', 1.0)  # Default to 1%
        self.entry_type = data.get('entry_type', 'market')  # 'market' or 'limit'
        self.notes = data.get('notes')
        self.expiry = data.get('expiry')  # Signal expiry timestamp
        self.meta = data.get('meta', {})
        
    def is_valid(self) -> bool:
        """Check if the signal has all required fields"""
        return all([
            self.symbol is not None,
            self.signal is not None,
            self.strategy is not None
        ])
    
    def is_expired(self) -> bool:
        """Check if the signal has expired"""
        if not self.expiry:
            return False
        
        try:
            expiry_dt = datetime.fromisoformat(self.expiry)
            return datetime.now() > expiry_dt
        except (ValueError, TypeError):
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'symbol': self.symbol,
            'strategy': self.strategy,
            'signal': self.signal,
            'timeframe': self.timeframe,
            'price': self.price,
            'timestamp': self.timestamp,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_percent': self.risk_percent,
            'entry_type': self.entry_type,
            'notes': self.notes,
            'expiry': self.expiry,
            'meta': self.meta
        }


class MarketContext:
    """
    Class for analyzing market context to make smarter trading decisions
    when processing TradingView signals
    """
    
    # Market regime types
    REGIME_BULLISH = "bullish"
    REGIME_BEARISH = "bearish"
    REGIME_SIDEWAYS = "sideways"
    REGIME_VOLATILE = "volatile"
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the market context analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.current_regime = self.REGIME_SIDEWAYS
        self.regime_confidence = 0.5
        self.vix_level = None
        self.market_open = False
        self.market_close_time = None
        self.last_update = None
        self.index_data = {}  # Cache for index data (SPY, QQQ, etc.)
        
        # Strategy risk multipliers
        self.strategy_risk_map = self.config.get('strategy_risk_map', {})
        
        # Default is full risk for all strategies
        self.default_strategy_multiplier = 1.0
        
    def update_market_data(self, tradier_client: TradierClient) -> None:
        """
        Update market context data from Tradier API
        
        Args:
            tradier_client: Initialized TradierClient instance
        """
        try:
            # Check if market is open
            clock_data = tradier_client.get_clock()
            if 'clock' in clock_data:
                self.market_open = clock_data['clock'].get('state') == 'open'
                if 'next_close' in clock_data['clock']:
                    next_close = clock_data['clock']['next_close']
                    self.market_close_time = datetime.fromisoformat(next_close.replace('Z', '+00:00'))
            
            # Get data for key market indices
            market_symbols = ['SPY', 'QQQ', 'VIX']
            quotes = tradier_client.get_quotes(market_symbols)
            
            # Extract VIX level
            if 'VIX' in quotes:
                self.vix_level = float(quotes['VIX'].get('last', 0))
            
            # Get daily data for SPY to determine market regime
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')
            
            spy_history = tradier_client.get_historical_data('SPY', 'daily', start_date, end_date)
            
            if spy_history and 'day' in spy_history:
                spy_days = spy_history['day']
                self._analyze_market_regime(spy_days)
            
            # Update timestamp
            self.last_update = datetime.now()
            
            logger.info(f"Market context updated: regime={self.current_regime}, VIX={self.vix_level}, "
                       f"market_open={self.market_open}")
                       
        except Exception as e:
            logger.error(f"Error updating market context: {str(e)}")
    
    def _analyze_market_regime(self, market_data: List[Dict[str, Any]]) -> None:
        """
        Analyze market data to determine current market regime
        
        Args:
            market_data: List of daily market data points
        """
        # Simple algorithm to determine market regime
        try:
            # Convert to pandas DataFrame for easier analysis
            df = pd.DataFrame(market_data)
            
            # Ensure numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Sort by date
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate basic indicators
            df['sma5'] = df['close'].rolling(5).mean()
            df['sma20'] = df['close'].rolling(20).mean()
            df['daily_return'] = df['close'].pct_change()
            df['atr'] = self._calculate_atr(df)
            df['atr_percent'] = df['atr'] / df['close'] * 100
            
            # Get most recent data
            last_row = df.iloc[-1]
            
            # Determine market regime
            if last_row['sma5'] > last_row['sma20'] and last_row['atr_percent'] < 1.5:
                self.current_regime = self.REGIME_BULLISH
                self.regime_confidence = 0.7
            elif last_row['sma5'] < last_row['sma20'] and last_row['atr_percent'] < 1.5:
                self.current_regime = self.REGIME_BEARISH
                self.regime_confidence = 0.7
            elif last_row['atr_percent'] > 2.0:
                self.current_regime = self.REGIME_VOLATILE
                self.regime_confidence = 0.8
            else:
                self.current_regime = self.REGIME_SIDEWAYS
                self.regime_confidence = 0.6
            
            # Consider VIX level if available
            if self.vix_level:
                if self.vix_level > 30:
                    # High VIX usually means volatile market
                    self.current_regime = self.REGIME_VOLATILE
                    self.regime_confidence = 0.9
                elif self.vix_level < 15 and self.current_regime == self.REGIME_BULLISH:
                    # Low VIX in bullish market increases confidence
                    self.regime_confidence = 0.8
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {str(e)}")
            self.current_regime = self.REGIME_SIDEWAYS
            self.regime_confidence = 0.5
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def get_position_size_multiplier(self, strategy: str, signal_type: str) -> float:
        """
        Get position size multiplier based on strategy and market context
        
        Args:
            strategy: Strategy name
            signal_type: Signal type ('buy', 'sell', etc.)
            
        Returns:
            Position size multiplier (0.0 to 1.0)
        """
        # Base multiplier starts at 1.0 (full position size)
        base_multiplier = 1.0
        
        # Adjust based on strategy-specific settings
        strategy_multiplier = self.strategy_risk_map.get(strategy, self.default_strategy_multiplier)
        
        # Adjust based on market regime
        regime_multiplier = 1.0
        
        if self.current_regime == self.REGIME_BULLISH:
            if signal_type in ['buy', 'long']:
                regime_multiplier = 1.0
            else:
                regime_multiplier = 0.5  # Reduce short positions in bullish market
        
        elif self.current_regime == self.REGIME_BEARISH:
            if signal_type in ['sell', 'short']:
                regime_multiplier = 1.0
            else:
                regime_multiplier = 0.5  # Reduce long positions in bearish market
        
        elif self.current_regime == self.REGIME_VOLATILE:
            regime_multiplier = 0.7  # Reduce position sizes in volatile markets
        
        elif self.current_regime == self.REGIME_SIDEWAYS:
            regime_multiplier = 0.8  # Slightly reduce position sizes in sideways markets
        
        # Adjust based on VIX
        vix_multiplier = 1.0
        if self.vix_level:
            if self.vix_level > 30:
                vix_multiplier = 0.6  # Significantly reduce position sizes when VIX is high
            elif self.vix_level > 20:
                vix_multiplier = 0.8  # Moderately reduce position sizes when VIX is elevated
        
        # Near market close, reduce position sizes
        time_multiplier = 1.0
        if self.market_open and self.market_close_time:
            minutes_to_close = (self.market_close_time - datetime.now()).total_seconds() / 60
            if minutes_to_close < 30:
                time_multiplier = 0.5  # Reduce position sizes in last 30 minutes
        
        # Combine all multipliers
        final_multiplier = base_multiplier * strategy_multiplier * regime_multiplier * vix_multiplier * time_multiplier
        
        # Ensure the multiplier is between 0 and 1
        final_multiplier = max(0.0, min(1.0, final_multiplier))
        
        logger.debug(f"Position size multiplier for {strategy} ({signal_type}): {final_multiplier:.2f} "
                    f"(regime={self.current_regime}, VIX={self.vix_level})")
        
        return final_multiplier
    
    def validate_signal(self, signal: TradingViewSignal) -> Dict[str, Any]:
        """
        Validate a TradingView signal against current market context
        
        Args:
            signal: TradingView signal to validate
            
        Returns:
            Dictionary with validation result and reason
        """
        # Basic validity check
        if not signal.is_valid():
            return {
                'valid': False,
                'reason': 'Missing required fields',
                'multiplier': 0.0
            }
        
        # Signal expiry check
        if signal.is_expired():
            return {
                'valid': False,
                'reason': 'Signal has expired',
                'multiplier': 0.0
            }
        
        # Market hours check
        if not self.market_open and signal.entry_type == 'market':
            return {
                'valid': False,
                'reason': 'Market is closed for market orders',
                'multiplier': 0.0
            }
        
        # Check for market close time
        if self.market_open and self.market_close_time:
            minutes_to_close = (self.market_close_time - datetime.now()).total_seconds() / 60
            if minutes_to_close < 10 and signal.entry_type == 'market':
                return {
                    'valid': False,
                    'reason': 'Too close to market close (< 10 minutes)',
                    'multiplier': 0.0
                }
        
        # Get position size multiplier
        multiplier = self.get_position_size_multiplier(signal.strategy, signal.signal)
        
        # Check if multiplier is too low
        if multiplier < 0.05:
            return {
                'valid': False,
                'reason': 'Position size multiplier too low',
                'multiplier': multiplier
            }
        
        # All checks passed
        return {
            'valid': True,
            'reason': 'Signal is valid',
            'multiplier': multiplier,
            'market_regime': self.current_regime,
            'vix_level': self.vix_level
        }


class WebhookHandler:
    """
    Handler for TradingView webhook signals
    
    This class processes webhook signals from TradingView, validates them
    against market context, and executes trades via the Tradier API.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the webhook handler
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_configuration()
        
        # Initialize Tradier client
        self._init_tradier_client()
        
        # Initialize market context
        self.market_context = MarketContext(self.config.get('market_context', {}))
        
        # Signal history
        self.signal_history = []
        self.max_history_size = self.config.get('max_signal_history', 100)
        
        # Thread-safe lock for signal processing
        self.processing_lock = threading.Lock()
        
        # Signal processing statistics
        self.stats = {
            'signals_received': 0,
            'signals_processed': 0,
            'signals_rejected': 0,
            'trades_executed': 0,
            'errors': 0
        }
        
        # Start background thread for market context updates
        self._start_background_updates()
        
        logger.info("Webhook handler initialized")
    
    def _init_tradier_client(self) -> None:
        """Initialize the Tradier client with API credentials"""
        # Get API credentials from config
        tradier_config = self.config.get('tradier', {})
        api_key = tradier_config.get('api_key')
        account_id = tradier_config.get('account_id')
        use_sandbox = tradier_config.get('use_sandbox', True)
        
        # Check for environment variables
        if not api_key:
            api_key = os.environ.get('TRADIER_API_KEY')
        
        if not account_id:
            account_id = os.environ.get('TRADIER_ACCOUNT_ID')
        
        if not api_key or not account_id:
            logger.warning("Tradier API credentials not found in config or environment variables")
            self.tradier_client = None
        else:
            self.tradier_client = TradierClient(
                api_key=api_key,
                account_id=account_id,
                sandbox=use_sandbox
            )
    
    def _start_background_updates(self) -> None:
        """Start background thread for market context updates"""
        def update_context():
            while True:
                try:
                    if self.tradier_client:
                        self.market_context.update_market_data(self.tradier_client)
                    
                    # Sleep for the configured interval
                    update_interval = self.config.get('market_context_update_interval', 300)  # 5 minutes default
                    time.sleep(update_interval)
                    
                except Exception as e:
                    logger.error(f"Error in market context update thread: {str(e)}")
                    time.sleep(60)  # Sleep for a minute on error
        
        # Start the thread
        context_thread = threading.Thread(target=update_context, daemon=True)
        context_thread.start()
        
        logger.info("Started background market context updates")
    
    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """
        Verify webhook signature using HMAC
        
        Args:
            payload: Raw request payload
            signature: Webhook signature from request headers
            
        Returns:
            True if signature is valid, False otherwise
        """
        # Get webhook secret from config
        webhook_secret = self.config.get('webhook_secret')
        
        if not webhook_secret or not signature:
            return False
        
        # Calculate expected signature
        expected_signature = hmac.new(
            webhook_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(signature, expected_signature)
    
    def process_webhook(self, payload: Dict[str, Any], headers: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Process a webhook from TradingView
        
        Args:
            payload: Webhook payload as dictionary
            headers: Request headers (optional)
            
        Returns:
            Dictionary with processing result
        """
        try:
            # Track statistics
            self.stats['signals_received'] += 1
            
            # Update timestamp
            received_at = datetime.now().isoformat()
            
            # Verify signature if headers provided and configured
            if headers and self.config.get('verify_webhook_signatures', False):
                signature = headers.get('X-Tradingview-Signature')
                
                # Convert payload back to bytes for verification
                payload_bytes = json.dumps(payload).encode('utf-8')
                
                if not self.verify_webhook_signature(payload_bytes, signature):
                    logger.warning("Invalid webhook signature")
                    self.stats['signals_rejected'] += 1
                    return {
                        'status': 'error',
                        'message': 'Invalid webhook signature',
                        'timestamp': received_at
                    }
            
            # Parse signal
            signal = TradingViewSignal(payload)
            
            # Save to history
            signal_record = {
                'signal': signal.to_dict(),
                'received_at': received_at,
                'processed': False,
                'result': None
            }
            
            self.signal_history.append(signal_record)
            
            # Trim history if needed
            if len(self.signal_history) > self.max_history_size:
                self.signal_history = self.signal_history[-self.max_history_size:]
            
            # Check if Tradier client is initialized
            if not self.tradier_client:
                logger.error("Tradier client not initialized")
                signal_record['result'] = {
                    'status': 'error',
                    'message': 'Tradier client not initialized'
                }
                self.stats['errors'] += 1
                return signal_record['result']
            
            # Process the signal in a thread-safe manner
            with self.processing_lock:
                # Validate signal against market context
                validation = self.market_context.validate_signal(signal)
                
                if not validation['valid']:
                    logger.info(f"Signal rejected: {validation['reason']} "
                                f"(Symbol: {signal.symbol}, Strategy: {signal.strategy})")
                    
                    signal_record['processed'] = True
                    signal_record['result'] = {
                        'status': 'rejected',
                        'message': validation['reason'],
                        'market_context': {
                            'regime': self.market_context.current_regime,
                            'vix': self.market_context.vix_level,
                            'market_open': self.market_context.market_open
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.stats['signals_rejected'] += 1
                    return signal_record['result']
                
                # Execute the signal
                execution_result = self._execute_signal(signal, validation)
                
                # Update signal record
                signal_record['processed'] = True
                signal_record['result'] = execution_result
                
                # Track statistics
                self.stats['signals_processed'] += 1
                if execution_result['status'] == 'success':
                    self.stats['trades_executed'] += 1
                elif execution_result['status'] == 'error':
                    self.stats['errors'] += 1
                
                return execution_result
                
        except Exception as e:
            error_msg = f"Error processing webhook: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'] += 1
            
            return {
                'status': 'error',
                'message': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    def _execute_signal(self, signal: TradingViewSignal, validation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trading signal via Tradier API
        
        Args:
            signal: Validated TradingView signal
            validation: Validation result with position size multiplier
            
        Returns:
            Dictionary with execution result
        """
        try:
            # Get account information
            account_info = self.tradier_client.get_account_balances()
            
            # Extract buying power
            buying_power = float(account_info.get('buying_power', 0))
            total_equity = float(account_info.get('total_equity', 0))
            
            # Check if we have enough buying power
            if buying_power <= 0:
                return {
                    'status': 'error',
                    'message': 'Insufficient buying power',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate position size
            multiplier = validation['multiplier']
            risk_percentage = signal.risk_percent * multiplier
            max_position_value = total_equity * (risk_percentage / 100)
            
            # Get current price if not provided
            price = signal.price
            if not price:
                quote = self.tradier_client.get_quote(signal.symbol)
                price = float(quote.get('last', 0))
                
                if price <= 0:
                    return {
                        'status': 'error',
                        'message': 'Could not determine current price',
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Calculate number of shares
            shares = int(max_position_value / price)
            
            # Ensure minimum number of shares
            min_shares = self.config.get('min_shares', 1)
            if shares < min_shares:
                if max_position_value >= price * min_shares:
                    shares = min_shares
                else:
                    return {
                        'status': 'error',
                        'message': f'Position size too small (less than {min_shares} shares)',
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Execute the trade based on signal type and entry type
            result = None
            
            # Map signal type to order side
            signal_map = {
                'buy': 'buy',
                'sell': 'sell',
                'long': 'buy',
                'short': 'sell_short',
                'close_long': 'sell',
                'close_short': 'buy_to_cover'
            }
            
            order_side = signal_map.get(signal.signal.lower())
            
            if not order_side:
                return {
                    'status': 'error',
                    'message': f'Unsupported signal type: {signal.signal}',
                    'timestamp': datetime.now().isoformat()
                }
            
            logger.info(f"Executing {order_side} order for {shares} shares of {signal.symbol} at {price} "
                       f"(Strategy: {signal.strategy}, Risk: {risk_percentage:.2f}%)")
            
            # Place the order
            if signal.entry_type.lower() == 'market':
                result = self.tradier_client.place_equity_order(
                    symbol=signal.symbol,
                    side=order_side,
                    quantity=shares,
                    order_type='market',
                    duration='day'
                )
            else:  # Limit order
                result = self.tradier_client.place_equity_order(
                    symbol=signal.symbol,
                    side=order_side,
                    quantity=shares,
                    order_type='limit',
                    price=price,
                    duration='day'
                )
            
            # Check if we need to place a stop loss
            if signal.stop_loss and order_side in ['buy', 'buy_to_cover']:
                # Calculate stop loss
                stop_price = signal.stop_loss
                
                # Place stop loss order
                stop_result = self.tradier_client.place_equity_order(
                    symbol=signal.symbol,
                    side='sell',
                    quantity=shares,
                    order_type='stop',
                    stop=stop_price,
                    duration='gtc'  # Good Till Cancelled
                )
                
                logger.info(f"Placed stop loss order for {shares} shares of {signal.symbol} at {stop_price}")
            
            # Check if we need to place a take profit
            if signal.take_profit and order_side in ['buy', 'buy_to_cover']:
                # Calculate take profit
                take_profit_price = signal.take_profit
                
                # Place take profit order
                take_profit_result = self.tradier_client.place_equity_order(
                    symbol=signal.symbol,
                    side='sell',
                    quantity=shares,
                    order_type='limit',
                    price=take_profit_price,
                    duration='gtc'  # Good Till Cancelled
                )
                
                logger.info(f"Placed take profit order for {shares} shares of {signal.symbol} at {take_profit_price}")
            
            # Return success result
            return {
                'status': 'success',
                'message': f'Order placed successfully (ID: {result.get("id")})',
                'order_id': result.get('id'),
                'symbol': signal.symbol,
                'side': order_side,
                'quantity': shares,
                'price': price,
                'strategy': signal.strategy,
                'risk_percent': risk_percentage,
                'multiplier': multiplier,
                'market_context': {
                    'regime': self.market_context.current_regime,
                    'vix': self.market_context.vix_level
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Error executing signal: {str(e)}"
            logger.error(error_msg)
            
            return {
                'status': 'error',
                'message': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_signals_history(self, limit: int = None, strategy: str = None, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Get signal history with optional filtering
        
        Args:
            limit: Maximum number of signals to return
            strategy: Filter by strategy name
            symbol: Filter by symbol
            
        Returns:
            List of signal records
        """
        filtered_history = self.signal_history
        
        # Filter by strategy if specified
        if strategy:
            filtered_history = [
                record for record in filtered_history 
                if record['signal']['strategy'] == strategy
            ]
        
        # Filter by symbol if specified
        if symbol:
            filtered_history = [
                record for record in filtered_history 
                if record['signal']['symbol'] == symbol
            ]
        
        # Sort by received timestamp (newest first)
        filtered_history = sorted(
            filtered_history,
            key=lambda x: x['received_at'],
            reverse=True
        )
        
        # Apply limit if specified
        if limit:
            filtered_history = filtered_history[:limit]
        
        return filtered_history
    
    def get_stats(self) -> Dict[str, Any]:
        """Get signal processing statistics"""
        return {
            **self.stats,
            'market_context': {
                'regime': self.market_context.current_regime,
                'vix': self.market_context.vix_level,
                'market_open': self.market_context.market_open,
                'regime_confidence': self.market_context.regime_confidence,
                'last_update': self.market_context.last_update.isoformat() if self.market_context.last_update else None
            }
        }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize webhook handler
    handler = WebhookHandler()
    
    # Example webhook payload from TradingView
    example_payload = {
        "symbol": "AAPL",
        "strategy": "rsi_oversold",
        "signal": "buy",
        "timeframe": "1h",
        "price": 150.25,
        "stop_loss": 148.50,
        "take_profit": 155.00,
        "risk": 1.0
    }
    
    # Process the webhook
    result = handler.process_webhook(example_payload)
    
    # Print the result
    print(f"Webhook result: {json.dumps(result, indent=2)}")
    
    # Get signal history
    history = handler.get_signals_history(limit=5)
    print(f"Recent signals: {json.dumps(history, indent=2)}")
    
    # Get statistics
    stats = handler.get_stats()
    print(f"Stats: {json.dumps(stats, indent=2)}") 