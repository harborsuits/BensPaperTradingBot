import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from trading_bot.market_context.context_analyzer import MarketContextAnalyzer
from trading_bot.brokers.tradier_client import TradierClient
from trading_bot.webhook_handler import MarketContext
from trading_bot.trading_strategy import TradingStrategy
from trading_bot.position_sizing import PositionSizer


class EnhancedBacktester:
    """
    Enhanced backtesting framework that tests the entire trading pipeline
    including market context analysis, signal generation, and order execution.
    """
    
    def __init__(
        self,
        strategy: TradingStrategy,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0,
        data_directory: str = "data/historical",
        results_directory: str = "results/backtests",
        commission: float = 0.001,  # 0.1% commission
        slippage: float = 0.001,    # 0.1% slippage
        tradier_client: Optional[TradierClient] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the backtester with the given parameters.
        
        Args:
            strategy: Trading strategy to backtest
            symbols: List of symbols to include in the backtest
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            initial_capital: Initial capital for the backtest
            data_directory: Directory containing historical data
            results_directory: Directory to save backtest results
            commission: Commission rate per trade
            slippage: Slippage percentage per trade
            tradier_client: Tradier client for fetching historical data
            logger: Logger instance
        """
        self.strategy = strategy
        self.strategy_name = strategy.__class__.__name__
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Create directories if they don't exist
        self.data_directory = data_directory
        self.results_directory = results_directory
        os.makedirs(self.data_directory, exist_ok=True)
        os.makedirs(self.results_directory, exist_ok=True)
        
        # Initialize logger
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize market data provider
        self.tradier_client = tradier_client
        
        # Initialize market context analyzer
        self.context_analyzer = MarketContextAnalyzer({})
        
        # Initialize position sizer
        self.position_sizer = PositionSizer({"default_risk_per_trade": 0.01})
        
        # Results storage
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []
        self.market_contexts = []
        self.signals = []
        self.positions = []
        self.orders = []
        
        self.logger.info(f"Initialized backtester for {self.strategy_name} on {len(symbols)} symbols from {start_date} to {end_date}")
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for all symbols.
        
        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        data = {}
        
        if self.tradier_client:
            for symbol in self.symbols:
                try:
                    # Convert dates to datetime objects if they're strings
                    start_date = pd.to_datetime(self.start_date).strftime("%Y-%m-%d")
                    end_date = pd.to_datetime(self.end_date).strftime("%Y-%m-%d")
                    
                    # Fetch data using Tradier client
                    historical_data = self.tradier_client.get_historical_dataframe(
                        symbol=symbol, 
                        interval="daily",
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if not historical_data.empty:
                        historical_data.set_index('date', inplace=True)
                        data[symbol] = historical_data
                        self.logger.info(f"Loaded {len(historical_data)} days of data for {symbol}")
                    else:
                        self.logger.warning(f"No data retrieved for {symbol}")
                        
                except Exception as e:
                    self.logger.error(f"Error loading data for {symbol}: {str(e)}")
        else:
            # Try to load from CSV files if no Tradier client
            for symbol in self.symbols:
                try:
                    file_path = os.path.join(self.data_directory, f"{symbol}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path, index_col='date', parse_dates=True)
                        data[symbol] = df
                        self.logger.info(f"Loaded {len(df)} rows of data for {symbol} from file")
                    else:
                        self.logger.warning(f"No data file found for {symbol} at {file_path}")
                except Exception as e:
                    self.logger.error(f"Error loading data file for {symbol}: {str(e)}")
        
        if not data:
            self.logger.error("No data loaded for any symbols")
            
        return data
    
    def run(self) -> Dict[str, Any]:
        """
        Run the backtest for the specified period.
        
        Returns:
            Dictionary containing backtest results
        """
        self.logger.info(f"Starting backtest run for {self.strategy_name}")
        
        # Load historical data
        data = self.load_data()
        
        if not data:
            self.logger.error("Cannot run backtest without data")
            return {"error": "No data available for backtest"}
        
        # Get unique dates across all symbols
        all_dates = set()
        for symbol in self.symbols:
            if symbol in data:
                all_dates.update(data[symbol].index.tolist())
        trading_days = sorted(list(all_dates))
        
        if len(trading_days) == 0:
            self.logger.error("No trading days found in data")
            return {"error": "No trading days found in data"}
            
        self.logger.info(f"Backtesting over {len(trading_days)} trading days")
        
        # Initialize backtest variables
        current_capital = self.initial_capital
        current_positions = {symbol: 0 for symbol in self.symbols}
        current_price = {symbol: 0.0 for symbol in self.symbols}
        
        # Run backtest day by day
        for i, date in enumerate(trading_days):
            # Skip the first day as we need historical data for indicators
            if i == 0:
                continue
            
            # Get data up to current date for context analysis
            current_data = {}
            for symbol in self.symbols:
                if symbol in data and date in data[symbol].index:
                    current_data[symbol] = data[symbol].loc[:date]
                    current_price[symbol] = data[symbol].loc[date, 'close']
            
            # Skip if no data for this date
            if not current_data:
                continue
            
            # Analyze market context
            market_context = self._analyze_market_context(current_data, date)
            self.market_contexts.append({
                'date': date,
                'context': market_context
            })
            
            # Generate signals based on strategy and market context
            signals = self._generate_signals(current_data, market_context, date)
            self.signals.append({
                'date': date,
                'signals': signals
            })
            
            # Update positions based on signals
            new_positions, orders = self._update_positions(
                current_positions,
                signals,
                current_data,
                current_price,
                current_capital,
                market_context,
                date
            )
            
            self.positions.append({
                'date': date,
                'positions': new_positions.copy()
            })
            
            self.orders.extend(orders)
            
            # Update current positions
            current_positions = new_positions.copy()
            
            # Calculate current capital based on trades
            for order in orders:
                if order['side'] == 'buy':
                    current_capital -= order['cost'] + order['commission'] + order['slippage']
                else:  # sell
                    current_capital += order['cost'] - order['commission'] - order['slippage']
            
            # Calculate portfolio value at end of day
            portfolio_value = current_capital
            for symbol, quantity in current_positions.items():
                if symbol in current_price and quantity > 0:
                    portfolio_value += current_price[symbol] * quantity
            
            self.equity_curve.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': current_capital
            })
            
            # Calculate daily return
            previous_value = self.equity_curve[-2]['portfolio_value'] if i > 1 else self.initial_capital
            daily_return = (portfolio_value - previous_value) / previous_value
            self.daily_returns.append({
                'date': date,
                'return': daily_return
            })
            
            if i % 20 == 0:  # Log progress every 20 days
                self.logger.info(f"Processed {i}/{len(trading_days)} days. Current portfolio value: ${portfolio_value:.2f}")
        
        self.logger.info(f"Backtest completed. Final portfolio value: ${self.equity_curve[-1]['portfolio_value']:.2f}")
        
        # Prepare results
        results = self._prepare_results()
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _analyze_market_context(self, data: Dict[str, pd.DataFrame], date) -> Dict[str, Any]:
        """
        Analyze market context based on historical data.
        
        Args:
            data: Dictionary mapping symbols to their historical data
            date: Current date being processed
            
        Returns:
            Dictionary containing market context information
        """
        # Initialize a basic market context
        context = {
            "date": date,
            "regime_type": "unknown",
            "volatility": 0.0,
            "trend_strength": 0.0,
            "market_sentiment": 0.0
        }
        
        try:
            # Focus on SPY if available for market context
            spy_symbol = "SPY"
            if spy_symbol in data and len(data[spy_symbol]) >= 20:
                spy_data = data[spy_symbol]
                
                # Calculate volatility (20-day ATR as percentage of price)
                if len(spy_data) >= 20:
                    high = spy_data['high'].values
                    low = spy_data['low'].values
                    close = spy_data['close'].values
                    
                    tr1 = np.abs(high[1:] - low[1:])
                    tr2 = np.abs(high[1:] - close[:-1])
                    tr3 = np.abs(low[1:] - close[:-1])
                    
                    tr = np.max(np.vstack([tr1, tr2, tr3]), axis=0)
                    atr = np.mean(tr[-20:])
                    
                    current_price = spy_data['close'].iloc[-1]
                    volatility = atr / current_price
                    context["volatility"] = volatility
                    
                    # Determine regime type based on moving averages and volatility
                    if len(spy_data) >= 50:
                        ma20 = spy_data['close'].rolling(20).mean()
                        ma50 = spy_data['close'].rolling(50).mean()
                        
                        current_ma20 = ma20.iloc[-1]
                        current_ma50 = ma50.iloc[-1]
                        
                        # Calculate trend strength (ratio of recent return to volatility)
                        returns_20d = (spy_data['close'].iloc[-1] / spy_data['close'].iloc[-21] - 1)
                        trend_strength = returns_20d / volatility if volatility > 0 else 0
                        context["trend_strength"] = trend_strength
                        
                        # Determine regime
                        if current_ma20 > current_ma50 and volatility < 0.015:
                            context["regime_type"] = "bullish"
                        elif current_ma20 < current_ma50 and volatility < 0.015:
                            context["regime_type"] = "bearish"
                        elif volatility > 0.02:
                            context["regime_type"] = "volatile"
                        else:
                            context["regime_type"] = "sideways"
                
                # Calculate market sentiment based on recent returns
                if len(spy_data) >= 5:
                    recent_returns = spy_data['close'].pct_change().iloc[-5:]
                    sentiment = np.mean(recent_returns) * 20  # Scale to reasonable range
                    context["market_sentiment"] = sentiment
            
        except Exception as e:
            self.logger.error(f"Error analyzing market context: {str(e)}")
        
        return context
    
    def _generate_signals(self, data: Dict[str, pd.DataFrame], market_context: Dict[str, Any], date) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on strategy and market context.
        
        Args:
            data: Dictionary mapping symbols to their historical data
            market_context: Current market context
            date: Current date being processed
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        try:
            for symbol in self.symbols:
                if symbol in data:
                    symbol_data = data[symbol]
                    
                    # Use the strategy to generate signals
                    strategy_signals = self.strategy.generate_signals(symbol_data)
                    
                    # Filter for signals on the current date
                    if isinstance(strategy_signals, pd.DataFrame):
                        current_signals = strategy_signals[strategy_signals.index == date]
                        
                        for idx, row in current_signals.iterrows():
                            if 'signal' in row and row['signal'] != 0:
                                signal_type = 'buy' if row['signal'] > 0 else 'sell'
                                signals.append({
                                    'symbol': symbol,
                                    'date': date,
                                    'signal': signal_type,
                                    'strength': abs(row['signal']),
                                    'price': data[symbol].loc[date, 'close'],
                                    'strategy': self.strategy_name
                                })
                    else:
                        # Handle dict-based signals
                        for sig in strategy_signals:
                            if sig.get('date') == date and sig.get('symbol') == symbol:
                                signals.append(sig)
                                
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
        
        return signals
    
    def _update_positions(
        self, 
        current_positions: Dict[str, int],
        signals: List[Dict[str, Any]],
        data: Dict[str, pd.DataFrame],
        current_prices: Dict[str, float],
        available_capital: float,
        market_context: Dict[str, Any],
        date
    ) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        """
        Update positions based on signals and available capital.
        
        Args:
            current_positions: Current positions (symbol -> quantity)
            signals: Trading signals
            data: Historical data
            current_prices: Current prices for each symbol
            available_capital: Available capital for new positions
            market_context: Current market context
            date: Current date
            
        Returns:
            Tuple of (new positions, orders executed)
        """
        new_positions = current_positions.copy()
        orders = []
        
        # Create a context object for position sizing
        context = MarketContext({})
        context.current_regime = market_context.get('regime_type', 'sideways')
        context.regime_confidence = 0.5
        context.vix_level = market_context.get('volatility', 0.015) * 100  # Convert to percentage
        
        for signal in signals:
            symbol = signal['symbol']
            signal_type = signal['signal']
            price = signal.get('price', current_prices.get(symbol, 0))
            
            if price <= 0:
                continue
                
            if symbol not in new_positions:
                new_positions[symbol] = 0
                
            # Apply context-based position sizing
            if signal_type == 'buy' and new_positions[symbol] <= 0:
                # Calculate position size based on risk and market context
                risk_multiplier = context.get_position_size_multiplier(self.strategy_name, signal_type)
                
                # Calculate shares based on risk percentage
                risk_pct = 0.01 * risk_multiplier  # 1% risk adjusted by market context
                position_value = available_capital * risk_pct
                shares = int(position_value / price)
                
                if shares > 0 and price * shares <= available_capital:
                    # Create buy order
                    commission = price * shares * self.commission
                    slippage = price * shares * self.slippage
                    
                    order = {
                        'date': date,
                        'symbol': symbol,
                        'side': 'buy',
                        'quantity': shares,
                        'price': price,
                        'cost': price * shares,
                        'commission': commission,
                        'slippage': slippage,
                        'total_cost': price * shares + commission + slippage
                    }
                    
                    orders.append(order)
                    new_positions[symbol] += shares
                    
            elif signal_type == 'sell' and new_positions[symbol] > 0:
                # Sell existing position
                shares = new_positions[symbol]
                
                # Create sell order
                commission = price * shares * self.commission
                slippage = price * shares * self.slippage
                
                order = {
                    'date': date,
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': shares,
                    'price': price,
                    'cost': price * shares,
                    'commission': commission,
                    'slippage': slippage,
                    'total_cost': commission + slippage
                }
                
                orders.append(order)
                new_positions[symbol] = 0
        
        return new_positions, orders
    
    def _prepare_results(self) -> Dict[str, Any]:
        """
        Prepare final backtest results.
        
        Returns:
            Dictionary containing comprehensive backtest results
        """
        # Convert lists to DataFrames
        equity_curve_df = pd.DataFrame(self.equity_curve)
        if not equity_curve_df.empty:
            equity_curve_df.set_index('date', inplace=True)
        
        daily_returns_df = pd.DataFrame(self.daily_returns)
        if not daily_returns_df.empty:
            daily_returns_df.set_index('date', inplace=True)
        
        # Extract trades from orders
        trades_df = pd.DataFrame(self.orders)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            daily_returns_df['return'].values if not daily_returns_df.empty else np.array([]),
            self.equity_curve[0]['portfolio_value'] if self.equity_curve else self.initial_capital,
            self.equity_curve[-1]['portfolio_value'] if self.equity_curve else self.initial_capital
        )
        
        # Prepare final results dictionary
        results = {
            'strategy_name': self.strategy_name,
            'symbols': self.symbols,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'final_portfolio_value': self.equity_curve[-1]['portfolio_value'] if self.equity_curve else self.initial_capital,
            'total_return': (self.equity_curve[-1]['portfolio_value'] - self.initial_capital) / self.initial_capital if self.equity_curve else 0,
            'equity_curve': equity_curve_df,
            'daily_returns': daily_returns_df,
            'performance_metrics': performance_metrics,
            'trades': trades_df,
            'market_contexts': self.market_contexts,
            'signals': self.signals,
            'final_positions': self.positions[-1]['positions'] if self.positions else {}
        }
        
        return results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Save backtest results to disk.
        
        Args:
            results: Dictionary containing backtest results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(
            self.results_directory,
            f"{self.strategy_name}_{timestamp}"
        )
        os.makedirs(result_dir, exist_ok=True)
        
        # Save basic information as JSON
        basic_info = {
            'strategy_name': results['strategy_name'],
            'symbols': results['symbols'],
            'start_date': results['start_date'],
            'end_date': results['end_date'],
            'initial_capital': results['initial_capital'],
            'final_portfolio_value': results['final_portfolio_value'],
            'total_return': results['total_return'],
            'performance_metrics': results['performance_metrics']
        }
        
        with open(os.path.join(result_dir, 'summary.json'), 'w') as f:
            json.dump(basic_info, f, indent=4)
        
        # Save detailed dataframes as CSV
        if 'equity_curve' in results and not results['equity_curve'].empty:
            results['equity_curve'].to_csv(os.path.join(result_dir, 'equity_curve.csv'))
        
        if 'daily_returns' in results and not results['daily_returns'].empty:
            results['daily_returns'].to_csv(os.path.join(result_dir, 'daily_returns.csv'))
        
        if 'trades' in results and not results['trades'].empty:
            results['trades'].to_csv(os.path.join(result_dir, 'trades.csv'))
        
        self.logger.info(f"Backtest results saved to {result_dir}")
        return result_dir
    
    def _calculate_performance_metrics(
        self,
        daily_returns: np.ndarray,
        initial_capital: float,
        final_capital: float,
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        trading_days_per_year: int = 252
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics from daily returns.
        
        Args:
            daily_returns: Array of daily returns
            initial_capital: Initial capital
            final_capital: Final capital
            risk_free_rate: Annual risk-free rate
            trading_days_per_year: Number of trading days per year
        
        Returns:
            Dictionary containing performance metrics
        """
        # Basic return metrics
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Handling empty returns array
        if len(daily_returns) == 0:
            return {
                'total_return': total_return,
                'annual_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_return': 0,
                'volatility': 0,
                'calmar_ratio': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_win': 0,
                'max_loss': 0
            }
        
        # Period metrics
        period_years = len(daily_returns) / trading_days_per_year
        annual_return = (1 + total_return) ** (1 / period_years) - 1 if period_years > 0 else 0
        
        # Calculate metrics that require non-empty returns
        avg_daily_return = np.mean(daily_returns)
        daily_volatility = np.std(daily_returns)
        annual_volatility = daily_volatility * np.sqrt(trading_days_per_year)
        
        # Sharpe ratio
        excess_returns = daily_returns - risk_free_rate / trading_days_per_year
        sharpe_ratio = np.sqrt(trading_days_per_year) * np.mean(excess_returns) / daily_volatility if daily_volatility > 0 else 0
        
        # Sortino ratio (downside risk)
        downside_returns = daily_returns[daily_returns < 0]
        downside_volatility = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = np.sqrt(trading_days_per_year) * avg_daily_return / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + daily_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # Win rate and profit factor
        winning_days = daily_returns > 0
        losing_days = daily_returns < 0
        win_rate = np.sum(winning_days) / len(daily_returns) if len(daily_returns) > 0 else 0
        
        gross_wins = np.sum(daily_returns[winning_days]) if np.sum(winning_days) > 0 else 0
        gross_losses = abs(np.sum(daily_returns[losing_days])) if np.sum(losing_days) > 0 else 0
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
        
        # Average win/loss
        avg_win = np.mean(daily_returns[winning_days]) if np.sum(winning_days) > 0 else 0
        avg_loss = np.mean(daily_returns[losing_days]) if np.sum(losing_days) > 0 else 0
        
        # Max win/loss
        max_win = np.max(daily_returns) if len(daily_returns) > 0 else 0
        max_loss = np.min(daily_returns) if len(daily_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_return': avg_daily_return,
            'volatility': annual_volatility,
            'calmar_ratio': calmar_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': max_win,
            'max_loss': max_loss
        } 