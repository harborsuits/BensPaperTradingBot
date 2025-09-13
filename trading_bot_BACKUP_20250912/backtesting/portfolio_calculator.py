import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortfolioCalculator:
    """
    Advanced portfolio calculator that accurately tracks portfolio value based on actual trades.
    
    This class provides:
    1. Accurate portfolio valuation based on FIFO accounting for entry/exit
    2. Support for various asset classes with different valuation logic
    3. Cash balance management including fees, interest, and dividends
    4. Multiple currency handling with FX rate adjustments
    5. Position grouping by strategy for attribution analysis
    6. Detailed trade-level metrics and portfolio-level aggregations
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        base_currency: str = 'USD',
        track_dividends: bool = True,
        track_interest: bool = True,
        track_fees: bool = True,
        track_slippage: bool = True,
        include_open_positions: bool = True,
        valuation_method: str = 'mark_to_market'  # or 'last_trade'
    ):
        """
        Initialize the portfolio calculator.
        
        Args:
            initial_capital: Starting capital in base currency
            base_currency: Base currency for calculations (e.g., 'USD')
            track_dividends: Whether to track dividend income
            track_interest: Whether to track interest income/expense
            track_fees: Whether to track transaction fees
            track_slippage: Whether to track slippage costs
            include_open_positions: Whether to include open positions in valuations
            valuation_method: Method for valuing portfolio ('mark_to_market' or 'last_trade')
        """
        self.initial_capital = initial_capital
        self.base_currency = base_currency
        self.track_dividends = track_dividends
        self.track_interest = track_interest
        self.track_fees = track_fees
        self.track_slippage = track_slippage
        self.include_open_positions = include_open_positions
        self.valuation_method = valuation_method
        
        # State tracking
        self.cash_balance = initial_capital
        self.current_positions = {}  # Symbol -> position details
        self.closed_positions = []  # List of closed position details
        self.open_trades = {}  # Symbol -> list of open trades (for FIFO accounting)
        self.cash_transactions = []  # Record of all cash transactions
        self.dividends = []  # Record of all dividends
        self.interest = []  # Record of all interest payments
        self.fees = []  # Record of all fees
        
        # Result tracking
        self.daily_portfolio_values = []  # Daily snapshot of portfolio value
        self.portfolio_snapshots = []  # Detailed portfolio snapshots
        
        # Strategy tracking for attribution
        self.strategy_allocations = {}  # Strategy -> allocation percentage
        self.strategy_performance = {}  # Strategy -> performance metrics
        
        # Asset class tracking
        self.asset_class_values = {}  # Asset class -> current value
        
        # FX rate cache
        self.fx_rates = {}  # Currency pair -> rate
        
        logger.info(f"Initialized PortfolioCalculator with {initial_capital} {base_currency}")

    def process_trades(self, trades: List[Dict[str, Any]]) -> None:
        """
        Process a list of trades to update portfolio state.
        
        Args:
            trades: List of trade dictionaries with details like symbol, quantity, price, etc.
        """
        for trade in trades:
            self._process_single_trade(trade)
        
        logger.info(f"Processed {len(trades)} trades")
    
    def _process_single_trade(self, trade: Dict[str, Any]) -> None:
        """
        Process a single trade and update portfolio state.
        
        Args:
            trade: Dictionary with trade details
        """
        # Extract trade details
        symbol = trade.get('symbol', '')
        quantity = float(trade.get('quantity', 0))
        price = float(trade.get('price', 0))
        direction = trade.get('direction', '').lower()  # 'buy' or 'sell'
        trade_date = trade.get('date', datetime.now())
        trade_type = trade.get('type', 'stock')  # 'stock', 'option', 'future', etc.
        strategy = trade.get('strategy', 'unknown')
        asset_class = trade.get('asset_class', 'equity')
        
        # Extract optional details
        currency = trade.get('currency', self.base_currency)
        commission = float(trade.get('commission', 0))
        slippage = float(trade.get('slippage', 0))
        trade_id = trade.get('id', f"{symbol}_{trade_date}_{direction}")
        
        # Calculate trade value in local currency
        trade_value = price * quantity
        
        # Convert to base currency if needed
        if currency != self.base_currency:
            fx_rate = self._get_fx_rate(currency, self.base_currency, trade_date)
            trade_value = trade_value * fx_rate
            commission = commission * fx_rate
            slippage = slippage * fx_rate
        
        # Calculate total cost including fees
        total_cost = trade_value + commission + slippage
        
        # Update cash balance
        if direction == 'buy':
            self.cash_balance -= total_cost
            self._record_cash_transaction(
                amount=-total_cost,
                transaction_type='buy',
                symbol=symbol,
                date=trade_date,
                details=f"Buy {quantity} of {symbol} @ {price}"
            )
        else:  # sell
            self.cash_balance += trade_value - commission - slippage
            self._record_cash_transaction(
                amount=trade_value - commission - slippage,
                transaction_type='sell',
                symbol=symbol,
                date=trade_date,
                details=f"Sell {quantity} of {symbol} @ {price}"
            )
        
        # Record fees if tracking
        if self.track_fees and commission > 0:
            self._record_fee(
                amount=commission,
                fee_type='commission',
                symbol=symbol,
                date=trade_date
            )
        
        if self.track_slippage and slippage > 0:
            self._record_fee(
                amount=slippage,
                fee_type='slippage',
                symbol=symbol,
                date=trade_date
            )
        
        # Update positions using FIFO accounting
        self._update_positions(
            symbol=symbol,
            quantity=quantity,
            direction=direction,
            price=price,
            trade_date=trade_date,
            trade_type=trade_type,
            strategy=strategy,
            asset_class=asset_class,
            trade_id=trade_id
        )
    
    def _update_positions(
        self,
        symbol: str,
        quantity: float,
        direction: str,
        price: float,
        trade_date: datetime,
        trade_type: str,
        strategy: str,
        asset_class: str,
        trade_id: str
    ) -> None:
        """
        Update portfolio positions based on a trade.
        
        Args:
            symbol: Security symbol
            quantity: Quantity traded
            direction: 'buy' or 'sell'
            price: Trade price
            trade_date: Date of the trade
            trade_type: Type of security traded
            strategy: Strategy that generated the trade
            asset_class: Asset class of the security
            trade_id: Unique ID for this trade
        """
        if direction == 'buy':
            # Create a new trade record
            trade_record = {
                'trade_id': trade_id,
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'date': trade_date,
                'trade_type': trade_type,
                'strategy': strategy,
                'asset_class': asset_class,
                'remaining_quantity': quantity
            }
            
            # Add to open trades
            if symbol not in self.open_trades:
                self.open_trades[symbol] = []
            
            self.open_trades[symbol].append(trade_record)
            
            # Update current positions summary
            if symbol not in self.current_positions:
                self.current_positions[symbol] = {
                    'quantity': 0,
                    'average_price': 0,
                    'cost_basis': 0,
                    'trade_type': trade_type,
                    'strategy': strategy,
                    'asset_class': asset_class,
                    'open_date': trade_date
                }
            
            # Update average price and cost basis
            position = self.current_positions[symbol]
            new_quantity = position['quantity'] + quantity
            
            if new_quantity > 0:
                # Weighted average calculation
                position['average_price'] = (
                    (position['average_price'] * position['quantity']) + (price * quantity)
                ) / new_quantity
                position['cost_basis'] += price * quantity
            
            position['quantity'] = new_quantity
            
        else:  # sell
            # Check if we have open trades for this symbol
            if symbol not in self.open_trades or not self.open_trades[symbol]:
                logger.warning(f"Attempted to sell {symbol} but no open trades found")
                return
            
            # Apply FIFO accounting
            remaining_to_sell = quantity
            realized_pnl = 0
            
            closed_trades = []
            
            for i, trade in enumerate(self.open_trades[symbol]):
                if remaining_to_sell <= 0:
                    break
                
                # How much we can sell from this trade
                sell_quantity = min(trade['remaining_quantity'], remaining_to_sell)
                
                # Calculate P&L for this portion
                trade_pnl = sell_quantity * (price - trade['price'])
                realized_pnl += trade_pnl
                
                # Update remaining quantities
                trade['remaining_quantity'] -= sell_quantity
                remaining_to_sell -= sell_quantity
                
                # If trade is fully closed, mark for recording
                if trade['remaining_quantity'] <= 0:
                    closed_trade = trade.copy()
                    closed_trade['exit_price'] = price
                    closed_trade['exit_date'] = trade_date
                    closed_trade['realized_pnl'] = trade_pnl
                    closed_trade['holding_period'] = (trade_date - trade['date']).days
                    
                    closed_trades.append(closed_trade)
            
            # Remove fully closed trades
            self.open_trades[symbol] = [
                trade for trade in self.open_trades[symbol] 
                if trade['remaining_quantity'] > 0
            ]
            
            # Record closed trades
            for trade in closed_trades:
                self.closed_positions.append(trade)
            
            # Update current position
            if symbol in self.current_positions:
                position = self.current_positions[symbol]
                position['quantity'] -= quantity
                
                # If position is closed, remove it
                if position['quantity'] <= 0:
                    del self.current_positions[symbol]
            
            # Record realized P&L
            if realized_pnl != 0:
                logger.info(f"Realized P&L of {realized_pnl:.2f} for {symbol}")
    
    def _get_fx_rate(
        self,
        from_currency: str,
        to_currency: str,
        date: datetime
    ) -> float:
        """
        Get FX rate for currency conversion.
        
        Args:
            from_currency: Source currency
            to_currency: Target currency
            date: Date for rate lookup
            
        Returns:
            Exchange rate (from_currency to to_currency)
        """
        # Same currency, no conversion needed
        if from_currency == to_currency:
            return 1.0
        
        # Check cache
        key = f"{from_currency}_{to_currency}_{date.strftime('%Y-%m-%d')}"
        if key in self.fx_rates:
            return self.fx_rates[key]
        
        # In a real implementation, this would fetch from a data provider
        # For now, use some fixed rates for common pairs
        base_rates = {
            'USD_EUR': 0.85,
            'USD_GBP': 0.75,
            'USD_JPY': 110.0,
            'EUR_USD': 1.18,
            'GBP_USD': 1.33,
            'JPY_USD': 0.009
        }
        
        pair = f"{from_currency}_{to_currency}"
        reverse_pair = f"{to_currency}_{from_currency}"
        
        rate = 1.0  # Default for unknown pairs
        
        if pair in base_rates:
            rate = base_rates[pair]
        elif reverse_pair in base_rates:
            rate = 1.0 / base_rates[reverse_pair]
        
        # Cache the result
        self.fx_rates[key] = rate
        
        return rate
    
    def _record_cash_transaction(
        self,
        amount: float,
        transaction_type: str,
        symbol: str,
        date: datetime,
        details: str = ''
    ) -> None:
        """
        Record a cash transaction.
        
        Args:
            amount: Transaction amount (positive for inflow, negative for outflow)
            transaction_type: Type of transaction
            symbol: Related symbol (if applicable)
            date: Transaction date
            details: Additional details
        """
        transaction = {
            'date': date,
            'amount': amount,
            'type': transaction_type,
            'symbol': symbol,
            'details': details,
            'balance_after': self.cash_balance
        }
        
        self.cash_transactions.append(transaction)
    
    def _record_fee(
        self,
        amount: float,
        fee_type: str,
        symbol: str,
        date: datetime
    ) -> None:
        """
        Record a fee or cost.
        
        Args:
            amount: Fee amount
            fee_type: Type of fee
            symbol: Related symbol
            date: Fee date
        """
        fee = {
            'date': date,
            'amount': amount,
            'type': fee_type,
            'symbol': symbol
        }
        
        self.fees.append(fee)
    
    def record_dividend(
        self,
        symbol: str,
        amount: float,
        date: datetime,
        shares_held: float = None
    ) -> None:
        """
        Record a dividend payment.
        
        Args:
            symbol: Security paying the dividend
            amount: Dividend amount (total)
            date: Ex-dividend date
            shares_held: Number of shares held (if not provided, use current position)
        """
        if not self.track_dividends:
            return
        
        if shares_held is None and symbol in self.current_positions:
            shares_held = self.current_positions[symbol]['quantity']
        else:
            shares_held = shares_held or 0
        
        dividend = {
            'date': date,
            'symbol': symbol,
            'amount': amount,
            'shares_held': shares_held
        }
        
        self.dividends.append(dividend)
        
        # Update cash balance
        self.cash_balance += amount
        
        # Record as cash transaction
        self._record_cash_transaction(
            amount=amount,
            transaction_type='dividend',
            symbol=symbol,
            date=date,
            details=f"Dividend from {symbol}"
        )
        
        logger.info(f"Recorded dividend of {amount:.2f} for {symbol}")
    
    def record_interest(
        self,
        amount: float,
        date: datetime,
        interest_type: str = 'cash_balance'
    ) -> None:
        """
        Record interest payment or charge.
        
        Args:
            amount: Interest amount (positive for received, negative for paid)
            date: Interest date
            interest_type: Type of interest
        """
        if not self.track_interest:
            return
        
        interest = {
            'date': date,
            'amount': amount,
            'type': interest_type
        }
        
        self.interest.append(interest)
        
        # Update cash balance
        self.cash_balance += amount
        
        # Record as cash transaction
        transaction_type = 'interest_received' if amount > 0 else 'interest_paid'
        
        self._record_cash_transaction(
            amount=amount,
            transaction_type=transaction_type,
            symbol='',
            date=date,
            details=f"{interest_type} interest"
        )
    
    def calculate_portfolio_value(
        self,
        date: datetime,
        market_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Calculate the current portfolio value based on positions and market data.
        
        Args:
            date: Valuation date
            market_data: Dictionary of market data (symbol -> data values)
            
        Returns:
            Dictionary with portfolio value details
        """
        cash_value = self.cash_balance
        position_value = 0.0
        positions_detail = {}
        asset_class_values = {}
        strategy_values = {}
        
        # Calculate value of open positions
        if self.include_open_positions:
            for symbol, position in self.current_positions.items():
                # Skip if position quantity is zero
                if position['quantity'] == 0:
                    continue
                    
                # Get current market price with better fallback logic
                current_price = None
                if symbol in market_data:
                    if 'close' in market_data[symbol]:
                        current_price = market_data[symbol]['close']
                    elif 'price' in market_data[symbol]:
                        current_price = market_data[symbol]['price']
                    elif 'last' in market_data[symbol]:
                        current_price = market_data[symbol]['last']
                
                # If no price found, use last known price
                if current_price is None:
                    current_price = position['average_price']
                    logger.warning(f"No market data for {symbol}, using last known price {current_price}")
                
                # Calculate position value
                position_quantity = position['quantity']
                position_price = current_price
                
                # Special handling for different asset types
                asset_class = position.get('asset_class', 'equity')
                trade_type = position.get('trade_type', 'stock')
                
                # Apply multipliers for futures, options, etc.
                contract_multiplier = 1.0
                
                if trade_type == 'future':
                    # Get asset-specific multiplier or use default
                    contract_multiplier = position.get('contract_multiplier', 100.0)
                elif trade_type == 'option':
                    contract_multiplier = position.get('contract_multiplier', 100.0)
                elif trade_type == 'forex':
                    # For forex, we need to convert to base currency
                    contract_multiplier = position.get('contract_multiplier', 1.0)
                    # Apply forex-specific calculations if needed
                    if 'forex_rate' in market_data.get(self.base_currency, {}):
                        position_price *= market_data[self.base_currency]['forex_rate']
                elif trade_type == 'crypto':
                    # For crypto, handle different lot sizes
                    contract_multiplier = position.get('contract_multiplier', 1.0)
                
                # Calculate position direction modifier (1 for long, -1 for short)
                direction_modifier = 1.0
                if position.get('direction', 'long') == 'short':
                    direction_modifier = -1.0
                
                # Calculate current value of the position
                current_value = position_quantity * position_price * contract_multiplier * direction_modifier
                
                # For shorts, add collateral to position value (already deducted from cash)
                if position.get('direction', 'long') == 'short':
                    # We need to account for the collateral set aside for shorts
                    collateral = position.get('collateral', position_quantity * position['average_price'] * contract_multiplier)
                    position_value += collateral
                
                # Calculate unrealized P&L
                cost_basis = position['average_price'] * position_quantity * contract_multiplier
                
                # For long positions: current_value - cost_basis
                # For short positions: cost_basis - (current_value without direction_modifier)
                if position.get('direction', 'long') == 'long':
                    unrealized_pnl = current_value - cost_basis
                else:
                    # For shorts, we need to calculate P&L differently
                    unrealized_pnl = cost_basis - (position_quantity * position_price * contract_multiplier)
                
                # Update position value (for shorts, we've already added collateral)
                if position.get('direction', 'long') == 'long':
                    position_value += current_value
                else:
                    position_value += unrealized_pnl  # For shorts, we only add the P&L
                
                # Track by asset class
                if asset_class not in asset_class_values:
                    asset_class_values[asset_class] = 0.0
                asset_class_values[asset_class] += (current_value if position.get('direction', 'long') == 'long' else unrealized_pnl)
                
                # Track by strategy
                strategy = position.get('strategy', 'unknown')
                if strategy not in strategy_values:
                    strategy_values[strategy] = 0.0
                strategy_values[strategy] += (current_value if position.get('direction', 'long') == 'long' else unrealized_pnl)
                
                # Store position details
                positions_detail[symbol] = {
                    'quantity': position_quantity,
                    'price': position_price,
                    'value': current_value if position.get('direction', 'long') == 'long' else unrealized_pnl,
                    'cost_basis': cost_basis,
                    'unrealized_pnl': unrealized_pnl,
                    'asset_class': asset_class,
                    'trade_type': trade_type,
                    'strategy': strategy,
                    'direction': position.get('direction', 'long'),
                    'contract_multiplier': contract_multiplier
                }
        
        # Calculate total portfolio value
        total_value = cash_value + position_value
        
        # Calculate asset allocation percentages
        asset_allocation = {}
        for asset_class, value in asset_class_values.items():
            if total_value > 0:
                asset_allocation[asset_class] = (value / total_value) * 100.0
            else:
                asset_allocation[asset_class] = 0.0
        
        # Calculate strategy allocation percentages
        strategy_allocation = {}
        for strategy, value in strategy_values.items():
            if total_value > 0:
                strategy_allocation[strategy] = (value / total_value) * 100.0
            else:
                strategy_allocation[strategy] = 0.0
        
        # Store the snapshot
        snapshot = {
            'date': date,
            'total_value': total_value,
            'cash_value': cash_value,
            'position_value': position_value,
            'positions': positions_detail,
            'asset_allocation': asset_allocation,
            'strategy_allocation': strategy_allocation
        }
        
        # Update class state
        self.asset_class_values = asset_class_values
        self.strategy_allocations = strategy_allocation
        
        self.portfolio_snapshots.append(snapshot)
        
        # Simplified daily record
        daily_record = {
            'date': date,
            'value': total_value,
            'cash': cash_value,
            'positions': position_value
        }
        
        self.daily_portfolio_values.append(daily_record)
        
        return snapshot
    
    def get_performance_series(self) -> pd.DataFrame:
        """
        Get a time series of portfolio performance.
        
        Returns:
            DataFrame with dates and portfolio values
        """
        if not self.daily_portfolio_values:
            return pd.DataFrame()
        
        data = []
        for record in self.daily_portfolio_values:
            data.append({
                'date': record['date'],
                'portfolio_value': record['value'],
                'cash': record['cash'],
                'positions': record['positions']
            })
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            # Calculate daily returns
            df = df.sort_values('date')
            df['daily_return'] = df['portfolio_value'].pct_change()
            
            # Calculate cumulative returns
            df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
            
        return df
    
    def get_trade_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate trade-level performance metrics.
        
        Returns:
            Dictionary with aggregate trade metrics
        """
        if not self.closed_positions:
            return {'all': {'trades': 0}}
        
        # Group trades by strategy
        trades_by_strategy = {}
        all_trades = []
        
        for trade in self.closed_positions:
            strategy = trade.get('strategy', 'unknown')
            
            if strategy not in trades_by_strategy:
                trades_by_strategy[strategy] = []
            
            trades_by_strategy[strategy].append(trade)
            all_trades.append(trade)
        
        # Calculate metrics for all trades and by strategy
        result = {'all': self._calculate_trade_metrics(all_trades)}
        
        for strategy, trades in trades_by_strategy.items():
            result[strategy] = self._calculate_trade_metrics(trades)
        
        return result
    
    def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate performance metrics for a list of trades.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with trade metrics
        """
        if not trades:
            return {'trades': 0}
        
        # Basic trade metrics
        num_trades = len(trades)
        profitable_trades = [t for t in trades if t.get('realized_pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('realized_pnl', 0) <= 0]
        
        num_profitable = len(profitable_trades)
        num_losing = len(losing_trades)
        
        win_rate = num_profitable / num_trades if num_trades > 0 else 0
        
        # Profit metrics
        total_profit = sum(t.get('realized_pnl', 0) for t in profitable_trades)
        total_loss = sum(t.get('realized_pnl', 0) for t in losing_trades)
        
        avg_profit = total_profit / num_profitable if num_profitable > 0 else 0
        avg_loss = total_loss / num_losing if num_losing > 0 else 0
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Holding period
        avg_hold_days = np.mean([t.get('holding_period', 0) for t in trades])
        
        # Risk-adjusted metrics
        total_pnl = total_profit + total_loss
        pnl_std = np.std([t.get('realized_pnl', 0) for t in trades])
        
        sharpe_ratio = total_pnl / pnl_std if pnl_std > 0 else 0
        
        return {
            'trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'total_pnl': total_profit + total_loss,
            'avg_hold_days': avg_hold_days,
            'sharpe_ratio': sharpe_ratio
        }
    
    def get_attribution_analysis(self) -> Dict[str, Dict[str, float]]:
        """
        Perform attribution analysis to determine contribution by strategy and asset class.
        
        Returns:
            Dictionary with attribution results
        """
        if not self.portfolio_snapshots:
            return {}
        
        # Calculate strategy contributions
        strategy_contribution = {}
        
        # Group trades by strategy
        trades_by_strategy = {}
        
        for trade in self.closed_positions:
            strategy = trade.get('strategy', 'unknown')
            
            if strategy not in trades_by_strategy:
                trades_by_strategy[strategy] = []
            
            trades_by_strategy[strategy].append(trade)
        
        # Calculate contribution from realized trades
        for strategy, trades in trades_by_strategy.items():
            total_pnl = sum(t.get('realized_pnl', 0) for t in trades)
            
            if strategy not in strategy_contribution:
                strategy_contribution[strategy] = {
                    'realized_pnl': 0,
                    'unrealized_pnl': 0,
                    'total_pnl': 0,
                    'allocation': 0
                }
            
            strategy_contribution[strategy]['realized_pnl'] = total_pnl
        
        # Get latest portfolio snapshot for current allocations and unrealized P&L
        if self.portfolio_snapshots:
            latest = self.portfolio_snapshots[-1]
            
            # Update strategy allocations
            for strategy, allocation in latest.get('strategy_allocation', {}).items():
                if strategy not in strategy_contribution:
                    strategy_contribution[strategy] = {
                        'realized_pnl': 0,
                        'unrealized_pnl': 0,
                        'total_pnl': 0,
                        'allocation': 0
                    }
                
                strategy_contribution[strategy]['allocation'] = allocation
            
            # Calculate unrealized P&L by strategy
            for symbol, details in latest.get('positions', {}).items():
                strategy = details.get('strategy', 'unknown')
                unrealized_pnl = details.get('unrealized_pnl', 0)
                
                if strategy not in strategy_contribution:
                    strategy_contribution[strategy] = {
                        'realized_pnl': 0,
                        'unrealized_pnl': 0,
                        'total_pnl': 0,
                        'allocation': 0
                    }
                
                strategy_contribution[strategy]['unrealized_pnl'] += unrealized_pnl
        
        # Calculate total P&L for each strategy
        for strategy in strategy_contribution:
            realized = strategy_contribution[strategy]['realized_pnl']
            unrealized = strategy_contribution[strategy]['unrealized_pnl']
            strategy_contribution[strategy]['total_pnl'] = realized + unrealized
        
        return {
            'strategy_attribution': strategy_contribution
        }
    
    def get_drawdown_analysis(self) -> Dict[str, Any]:
        """
        Calculate drawdown metrics for the portfolio.
        
        Returns:
            Dictionary with drawdown metrics and series
        """
        df = self.get_performance_series()
        
        if df.empty:
            return {'max_drawdown': 0}
        
        # Calculate drawdown series
        df['peak'] = df['portfolio_value'].cummax()
        df['drawdown'] = (df['portfolio_value'] / df['peak'] - 1) * 100
        
        # Find maximum drawdown
        max_drawdown = df['drawdown'].min()
        
        # Find drawdown periods
        is_drawdown = df['drawdown'] < 0
        drawdown_periods = []
        
        if is_drawdown.any():
            # Create a group identifier that changes when drawdown starts/ends
            df['drawdown_group'] = (is_drawdown != is_drawdown.shift()).cumsum()
            
            # For each group where drawdown is true, get stats
            for group, group_df in df[is_drawdown].groupby('drawdown_group'):
                if len(group_df) > 0:
                    start_date = group_df['date'].iloc[0]
                    end_date = group_df['date'].iloc[-1]
                    max_dd = group_df['drawdown'].min()
                    duration = len(group_df)
                    
                    drawdown_periods.append({
                        'start_date': start_date,
                        'end_date': end_date,
                        'max_drawdown': max_dd,
                        'duration': duration
                    })
        
        # Find longest and deepest drawdowns
        if drawdown_periods:
            longest_dd = max(drawdown_periods, key=lambda x: x['duration'])
            deepest_dd = min(drawdown_periods, key=lambda x: x['max_drawdown'])
        else:
            longest_dd = {'duration': 0, 'max_drawdown': 0}
            deepest_dd = {'duration': 0, 'max_drawdown': 0}
        
        # Calculate time to recovery
        recovery_periods = []
        
        for period in drawdown_periods:
            end_date = period['end_date']
            max_dd = period['max_drawdown']
            
            # Find when portfolio recovered after this drawdown
            if end_date < df['date'].iloc[-1]:
                post_dd = df[df['date'] > end_date]
                recovery_idx = post_dd[post_dd['portfolio_value'] >= post_dd['peak'].iloc[0]].index
                
                if len(recovery_idx) > 0:
                    first_recovery = recovery_idx[0]
                    recovery_date = df.loc[first_recovery, 'date']
                    recovery_duration = (recovery_date - end_date).days
                    
                    recovery_periods.append({
                        'drawdown_end': end_date,
                        'recovery_date': recovery_date,
                        'recovery_duration': recovery_duration,
                        'max_drawdown': max_dd
                    })
        
        return {
            'max_drawdown': max_drawdown,
            'drawdown_periods': drawdown_periods,
            'longest_drawdown': longest_dd,
            'deepest_drawdown': deepest_dd,
            'recovery_periods': recovery_periods
        }
    
    def get_trade_level_metrics(self) -> pd.DataFrame:
        """
        Get detailed metrics for individual trades.
        
        Returns:
            DataFrame with trade-level metrics
        """
        if not self.closed_positions:
            return pd.DataFrame()
        
        # Extract relevant trade details
        trades_data = []
        
        for trade in self.closed_positions:
            trade_data = {
                'symbol': trade.get('symbol', ''),
                'strategy': trade.get('strategy', 'unknown'),
                'asset_class': trade.get('asset_class', 'equity'),
                'entry_date': trade.get('date', None),
                'exit_date': trade.get('exit_date', None),
                'entry_price': trade.get('price', 0),
                'exit_price': trade.get('exit_price', 0),
                'quantity': trade.get('quantity', 0),
                'pnl': trade.get('realized_pnl', 0),
                'pnl_pct': (trade.get('exit_price', 0) / trade.get('price', 1) - 1) * 100 if trade.get('price', 0) > 0 else 0,
                'holding_days': trade.get('holding_period', 0)
            }
            
            trades_data.append(trade_data)
        
        # Create DataFrame
        trades_df = pd.DataFrame(trades_data)
        
        if not trades_df.empty:
            # Calculate additional metrics
            trades_df['profitable'] = trades_df['pnl'] > 0
            
            # Calculate risk-reward ratio where possible
            trades_df['risk_reward'] = None
            
            # For real implementation, would need stop prices to calculate true risk-reward
        
        return trades_df 