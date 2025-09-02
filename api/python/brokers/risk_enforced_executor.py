import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

from trading_bot.brokers.tradier_client import TradierClient, TradierAPIError

# Import risk manager
from trading_bot.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)

class RiskEnforcedExecutor:
    """
    Trade execution module with enforced risk management
    
    Handles position sizing, risk checks, order execution, and trade management
    """
    
    def __init__(self, 
                tradier_client: TradierClient,
                risk_manager: RiskManager,
                max_position_pct: float = 0.05,
                max_risk_pct: float = 0.01,
                order_type: str = "market",
                order_duration: str = "day"):
        """
        Initialize the trade executor with risk management
        
        Args:
            tradier_client: Initialized Tradier client
            risk_manager: Risk manager instance for enforcing risk limits
            max_position_pct: Maximum position size as percentage of account equity
            max_risk_pct: Maximum risk per trade as percentage of account equity
            order_type: Default order type ('market', 'limit', 'stop', 'stop_limit')
            order_duration: Default order duration ('day', 'gtc')
        """
        self.client = tradier_client
        self.risk_manager = risk_manager
        self.max_position_pct = max_position_pct
        self.max_risk_pct = max_risk_pct
        self.default_order_type = order_type
        self.default_order_duration = order_duration
        
        # Trade history
        self.trade_history = []
        
        # Active trades
        self.active_trades = {}
        
        # Initialize with account data
        self.refresh_account_data()
        
        logger.info(f"RiskEnforcedExecutor initialized with max position: {max_position_pct:.1%}, max risk: {max_risk_pct:.1%}")
    
    def refresh_account_data(self):
        """Refresh account data from the broker"""
        try:
            # Get account summary
            account_summary = self.client.get_account_summary()
            
            # Extract key data
            self.account_equity = account_summary.get("equity", 0)
            self.cash_balance = account_summary.get("cash", 0)
            self.buying_power = account_summary.get("buying_power", 0)
            
            # Get existing positions
            self.positions = account_summary.get("positions", {}).get("details", [])
            
            # Calculate max position size in dollars
            self.max_position_dollars = self.account_equity * self.max_position_pct
            self.max_risk_dollars = self.account_equity * self.max_risk_pct
            
            logger.info(f"Account refreshed: equity=${self.account_equity:.2f}, "
                       f"max position=${self.max_position_dollars:.2f}, "
                       f"max risk=${self.max_risk_dollars:.2f}")
                       
            # Update active trades with current position data
            self._update_active_trades()
            
            # Update risk manager with fresh portfolio data
            self.risk_manager.update_portfolio_state(
                equity=self.account_equity,
                positions=self.positions,
                active_trades=self.active_trades
            )
            
        except Exception as e:
            logger.error(f"Error refreshing account data: {str(e)}")
            raise
    
    def _update_active_trades(self):
        """Update active trades with current position data"""
        # Map positions by symbol
        position_map = {pos.get("symbol"): pos for pos in self.positions}
        
        # Update active trades with current position data
        for trade_id, trade in self.active_trades.items():
            symbol = trade.get("symbol")
            if symbol in position_map:
                # Update trade with current position data
                position = position_map[symbol]
                trade["current_price"] = float(position.get("last_price", 0))
                trade["current_value"] = float(position.get("market_value", 0))
                trade["unrealized_pl"] = float(position.get("gain_loss", 0))
                trade["unrealized_pl_pct"] = float(position.get("gain_loss_percent", 0))
                trade["updated_at"] = datetime.now().isoformat()
            else:
                # Position has been closed
                if trade.get("status") == "open":
                    trade["status"] = "closed"
    
    def calculate_position_size(self, 
                               entry_price: float, 
                               stop_price: Optional[float] = None,
                               risk_pct: Optional[float] = None) -> Tuple[int, float]:
        """
        Calculate position size based on risk parameters
        
        Args:
            entry_price: Entry price per share
            stop_price: Stop loss price per share
            risk_pct: Risk percentage (overrides default max_risk_pct)
            
        Returns:
            Tuple of (shares_quantity, dollar_amount)
        """
        if not entry_price:
            logger.error("Entry price is required for position sizing")
            raise ValueError("Entry price is required for position sizing")
        
        # Use provided risk percentage or default
        risk_pct = risk_pct if risk_pct is not None else self.max_risk_pct
        risk_dollars = self.account_equity * risk_pct
        
        # If stop price is provided, calculate position size based on risk
        if stop_price and stop_price > 0:
            # Calculate risk per share
            if entry_price > stop_price:  # Long position
                risk_per_share = entry_price - stop_price
            else:  # Short position (should rarely happen)
                risk_per_share = stop_price - entry_price
            
            # Avoid division by zero
            if risk_per_share <= 0:
                logger.warning("Invalid risk per share, using default position sizing")
                shares = int(self.max_position_dollars / entry_price)
            else:
                # Calculate shares based on risk
                shares = int(risk_dollars / risk_per_share)
                
                # Check if calculated size exceeds max position
                if shares * entry_price > self.max_position_dollars:
                    logger.warning("Risk-based position size exceeds max position size, capping")
                    shares = int(self.max_position_dollars / entry_price)
        else:
            # Without stop, use maximum position size
            shares = int(self.max_position_dollars / entry_price)
        
        # Ensure at least 1 share
        shares = max(1, shares)
        
        # Calculate dollar amount
        dollar_amount = shares * entry_price
        
        logger.info(f"Position size calculated: {shares} shares (${dollar_amount:.2f})")
        
        return shares, dollar_amount
    
    def execute_trade(self, 
                     symbol: str, 
                     side: str, 
                     entry_price: Optional[float] = None,
                     stop_price: Optional[float] = None,
                     quantity: Optional[int] = None,
                     risk_pct: Optional[float] = None,
                     order_type: Optional[str] = None,
                     duration: Optional[str] = None,
                     strategy_name: Optional[str] = None,
                     metadata: Optional[Dict] = None) -> Dict:
        """
        Execute a trade with risk checks
        
        Args:
            symbol: Symbol to trade
            side: Trade side ('buy' or 'sell')
            entry_price: Limit price (None for market orders)
            stop_price: Stop loss price
            quantity: Number of shares (calculated from risk if not provided)
            risk_pct: Risk percentage (overrides default)
            order_type: Order type (overrides default)
            duration: Order duration (overrides default)
            strategy_name: Name of the strategy initiating the trade
            metadata: Additional trade metadata
            
        Returns:
            Trade details including ID and order information
        """
        try:
            # Refresh account data
            self.refresh_account_data()
            
            # Validate inputs
            if not symbol:
                raise ValueError("Symbol is required")
            
            if side not in ["buy", "sell"]:
                raise ValueError(f"Invalid side: {side}")
            
            # Get current price if not provided
            if not entry_price:
                quote = self.client.get_quote(symbol)
                entry_price = float(quote.get("last", 0))
                
                if entry_price <= 0:
                    raise ValueError(f"Invalid price for {symbol}: {entry_price}")
            
            # Calculate position size if not provided
            if quantity is None:
                quantity, dollar_amount = self.calculate_position_size(
                    entry_price=entry_price,
                    stop_price=stop_price,
                    risk_pct=risk_pct
                )
            else:
                dollar_amount = quantity * entry_price
            
            # Create order object for risk check
            order = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": entry_price,
                "stop_price": stop_price,
                "order_type": order_type or self.default_order_type,
                "dollar_amount": dollar_amount,
                "strategy": strategy_name,
                "metadata": metadata or {}
            }
            
            # *** CRITICAL NEW ADDITION: Risk check before execution ***
            # Check if this trade passes risk management rules
            risk_check_result = self.risk_manager.check_trade(order)
            
            if not risk_check_result["approved"]:
                logger.warning(f"Trade rejected by risk manager: {risk_check_result['reason']}")
                return {
                    "status": "rejected",
                    "reason": risk_check_result["reason"],
                    "trade_id": None,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": entry_price,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Proceed with order execution if approved
            # Use provided values or defaults
            order_type = order_type or self.default_order_type
            duration = duration or self.default_order_duration
            
            # Place the order
            order_result = self.client.place_equity_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                duration=duration,
                price=entry_price if order_type in ["limit", "stop_limit"] else None,
                stop=stop_price if order_type in ["stop", "stop_limit"] else None
            )
            
            # Generate a unique trade ID
            trade_id = str(uuid.uuid4())
            
            # Create trade record
            trade = {
                "trade_id": trade_id,
                "symbol": symbol,
                "side": side,
                "shares": quantity,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "order_type": order_type,
                "duration": duration,
                "status": "pending",
                "order_id": order_result.get("id"),
                "strategy": strategy_name,
                "metadata": metadata or {},
                "risk_pct": risk_pct or self.max_risk_pct,
                "dollar_amount": dollar_amount,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Store the trade
            self.active_trades[trade_id] = trade
            self.trade_history.append(trade)
            
            logger.info(f"Trade executed: {side} {quantity} {symbol} at ${entry_price:.2f}")
            
            # If stop price provided, place stop order
            if stop_price and stop_price > 0 and order_type not in ["stop", "stop_limit"]:
                stop_side = "sell" if side == "buy" else "buy"
                
                try:
                    stop_order_result = self.client.place_equity_order(
                        symbol=symbol,
                        side=stop_side,
                        quantity=quantity,
                        order_type="stop",
                        duration=duration,
                        stop=stop_price
                    )
                    
                    # Update trade with stop order info
                    trade["stop_order_id"] = stop_order_result.get("id")
                    
                    logger.info(f"Stop order placed: {stop_side} {quantity} {symbol} at ${stop_price:.2f}")
                    
                except Exception as stop_error:
                    logger.error(f"Error placing stop order: {str(stop_error)}")
                    trade["stop_order_error"] = str(stop_error)
            
            return trade
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            raise

    # The remaining methods (exit_trade, get_open_trades, etc.) remain essentially 
    # the same as in the original TradeExecutor class
