"""
Strategy Approval Workflow

This module provides a UI for strategy backtesting, approval, and deployment to paper trading.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union

# Import strategy components
from trading_bot.strategies.momentum import MomentumStrategy
from trading_bot.strategies.mean_reversion import MeanReversionStrategy
try:
    from trading_bot.strategies.trend_following import TrendFollowingStrategy
except ImportError:
    TrendFollowingStrategy = None
try:
    from trading_bot.strategies.volatility_breakout import VolatilityBreakout
except ImportError:
    VolatilityBreakout = None

# Configure logging
logger = logging.getLogger("strategy_approval")


class BacktestResult:
    """Holds the results of a strategy backtest"""
    
    def __init__(self, 
                 strategy_name: str, 
                 symbols: List[str],
                 start_date: str, 
                 end_date: str,
                 initial_capital: float,
                 params: Dict[str, Any]):
        self.strategy_name = strategy_name
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.params = params
        
        # Performance metrics
        self.final_equity = initial_capital
        self.returns_pct = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.trades_count = 0
        
        # Trade list and equity curve
        self.trades = []
        self.equity_curve = []
        
        # Approval status
        self.approved = False
        self.approved_date = None
        self.approved_by = None
        self.notes = ""
    
    def calculate_metrics(self, trades: List[Dict], equity_curve: List[float]):
        """Calculate performance metrics from trade list and equity curve"""
        self.trades = trades
        self.equity_curve = equity_curve
        
        if len(equity_curve) > 0:
            self.final_equity = equity_curve[-1]
            self.returns_pct = ((self.final_equity / self.initial_capital) - 1) * 100
            
            # Calculate drawdown
            max_equity = self.initial_capital
            max_drawdown = 0
            
            for equity in equity_curve:
                max_equity = max(max_equity, equity)
                drawdown = (max_equity - equity) / max_equity * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            self.max_drawdown = max_drawdown
        
        if len(trades) > 0:
            self.trades_count = len(trades)
            
            # Calculate win rate
            winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
            self.win_rate = len(winning_trades) / len(trades) * 100
            
            # Calculate profit factor
            gross_profit = sum([t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0])
            gross_loss = abs(sum([t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0]))
            
            if gross_loss > 0:
                self.profit_factor = gross_profit / gross_loss
            else:
                self.profit_factor = float('inf') if gross_profit > 0 else 0
            
            # Calculate Sharpe ratio (simplified)
            if len(equity_curve) > 1:
                daily_returns = [
                    (equity_curve[i] / equity_curve[i-1]) - 1 
                    for i in range(1, len(equity_curve))
                ]
                
                if len(daily_returns) > 0:
                    avg_return = sum(daily_returns) / len(daily_returns)
                    std_dev = (sum([(r - avg_return) ** 2 for r in daily_returns]) / len(daily_returns)) ** 0.5
                    
                    if std_dev > 0:
                        self.sharpe_ratio = (avg_return / std_dev) * (252 ** 0.5)  # Annualized
    
    def set_approval(self, approved: bool, approver: str, notes: str = ""):
        """Set approval status for the strategy backtest"""
        self.approved = approved
        self.approved_date = datetime.now().isoformat()
        self.approved_by = approver
        self.notes = notes
    
    def to_dict(self) -> Dict:
        """Convert backtest result to dictionary for serialization"""
        return {
            "strategy_name": self.strategy_name,
            "symbols": self.symbols,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "params": self.params,
            "performance": {
                "final_equity": self.final_equity,
                "returns_pct": self.returns_pct,
                "max_drawdown": self.max_drawdown,
                "sharpe_ratio": self.sharpe_ratio,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "trades_count": self.trades_count
            },
            "approval": {
                "approved": self.approved,
                "approved_date": self.approved_date,
                "approved_by": self.approved_by,
                "notes": self.notes
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BacktestResult':
        """Create backtest result object from dictionary"""
        result = cls(
            strategy_name=data.get("strategy_name", ""),
            symbols=data.get("symbols", []),
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date", ""),
            initial_capital=data.get("initial_capital", 10000),
            params=data.get("params", {})
        )
        
        # Set performance metrics
        performance = data.get("performance", {})
        result.final_equity = performance.get("final_equity", result.initial_capital)
        result.returns_pct = performance.get("returns_pct", 0.0)
        result.max_drawdown = performance.get("max_drawdown", 0.0)
        result.sharpe_ratio = performance.get("sharpe_ratio", 0.0)
        result.win_rate = performance.get("win_rate", 0.0)
        result.profit_factor = performance.get("profit_factor", 0.0)
        result.trades_count = performance.get("trades_count", 0)
        
        # Set approval status
        approval = data.get("approval", {})
        result.approved = approval.get("approved", False)
        result.approved_date = approval.get("approved_date", None)
        result.approved_by = approval.get("approved_by", None)
        result.notes = approval.get("notes", "")
        
        return result


class StrategyApprovalWorkflow:
    """
    Handles the workflow for strategy backtesting, approval, and deployment
    to paper trading.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the strategy approval workflow"""
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "approvals"
        )
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Available strategies
        self.strategies = {
            "Momentum": MomentumStrategy,
            "Mean Reversion": MeanReversionStrategy
        }
        
        # Add optional strategies if available
        if TrendFollowingStrategy:
            self.strategies["Trend Following"] = TrendFollowingStrategy
        
        if VolatilityBreakout:
            self.strategies["Volatility Breakout"] = VolatilityBreakout
        
        # Load previous backtest results
        self.backtest_results = self._load_backtest_results()
    
    def render(self):
        """Render the strategy approval workflow UI"""
        st.title("Strategy Backtesting & Approval")
        
        # Create tabs for different workflow stages
        tabs = st.tabs(["Configure & Backtest", "Review Results", "Approved Strategies"])
        
        with tabs[0]:
            self._render_backtest_tab()
        
        with tabs[1]:
            self._render_review_tab()
        
        with tabs[2]:
            self._render_approved_tab()
    
    def _render_backtest_tab(self):
        """Render the backtest configuration tab"""
        st.header("Configure Backtest")
        
        with st.form("backtest_config"):
            # Strategy selection
            strategy_name = st.selectbox(
                "Select Strategy",
                list(self.strategies.keys())
            )
            
            # Symbol selection
            symbols = st.multiselect(
                "Select Symbols",
                ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ", "META", "NVDA"],
                ["AAPL", "MSFT"]
            )
            
            # Date range
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=(datetime.now() - timedelta(days=365)).date()
                )
            
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now().date()
                )
            
            # Initial capital
            initial_capital = st.number_input(
                "Initial Capital",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000
            )
            
            # Strategy parameters
            st.subheader("Strategy Parameters")
            
            # Default parameters based on strategy
            if strategy_name == "Momentum":
                default_params = {
                    "rsi_period": 14,
                    "rsi_oversold": 30,
                    "rsi_overbought": 70,
                    "rate_of_change_period": 10
                }
            elif strategy_name == "Mean Reversion":
                default_params = {
                    "lookback_period": 20,
                    "std_dev_threshold": 2.0,
                    "ma_period": 50
                }
            elif strategy_name == "Trend Following":
                default_params = {
                    "fast_ma": 20,
                    "slow_ma": 50,
                    "signal_smoothing": 9
                }
            elif strategy_name == "Volatility Breakout":
                default_params = {
                    "atr_period": 14,
                    "breakout_multiplier": 1.5,
                    "stop_loss_atr": 2.0
                }
            else:
                default_params = {}
            
            # Render parameter inputs
            params = {}
            param_cols = st.columns(2)
            
            for i, (name, value) in enumerate(default_params.items()):
                with param_cols[i % 2]:
                    if isinstance(value, int):
                        params[name] = st.number_input(
                            f"{name.replace('_', ' ').title()}",
                            min_value=1,
                            max_value=1000,
                            value=value
                        )
                    elif isinstance(value, float):
                        params[name] = st.number_input(
                            f"{name.replace('_', ' ').title()}",
                            min_value=0.0,
                            max_value=100.0,
                            value=value,
                            format="%.2f"
                        )
            
            # Submit button
            submitted = st.form_submit_button("Run Backtest")
            
            if submitted:
                with st.spinner("Running backtest..."):
                    backtest_id = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Create backtest result
                    result = BacktestResult(
                        strategy_name=strategy_name,
                        symbols=symbols,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        initial_capital=initial_capital,
                        params=params
                    )
                    
                    # Run backtest
                    trades, equity_curve = self._run_backtest(
                        strategy_name=strategy_name,
                        symbols=symbols,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        initial_capital=initial_capital,
                        params=params
                    )
                    
                    # Calculate metrics
                    result.calculate_metrics(trades, equity_curve)
                    
                    # Save result
                    self.backtest_results[backtest_id] = result
                    self._save_backtest_results()
                    
                    st.success(f"Backtest completed: {result.returns_pct:.2f}% return, {result.trades_count} trades")
    
    def _render_review_tab(self):
        """Render the backtest review tab"""
        st.header("Review Backtest Results")
        
        # Get unapproved results
        unapproved_results = {
            id: result for id, result in self.backtest_results.items()
            if not result.approved
        }
        
        if not unapproved_results:
            st.info("No pending backtest results to review.")
            return
        
        # Select result to review
        result_id = st.selectbox(
            "Select Backtest",
            list(unapproved_results.keys()),
            format_func=lambda x: f"{unapproved_results[x].strategy_name} - {x.split('_')[1]}"
        )
        
        if not result_id:
            return
        
        result = unapproved_results[result_id]
        
        # Display result summary
        st.subheader(f"Strategy: {result.strategy_name}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Return", f"{result.returns_pct:.2f}%")
        with col2:
            st.metric("Max Drawdown", f"{result.max_drawdown:.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Win Rate", f"{result.win_rate:.2f}%")
        with col2:
            st.metric("Profit Factor", f"{result.profit_factor:.2f}")
        with col3:
            st.metric("Trades", str(result.trades_count))
        
        # Display equity curve
        if result.equity_curve:
            st.subheader("Equity Curve")
            
            # Create dates for x-axis
            start_date = datetime.strptime(result.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(result.end_date, "%Y-%m-%d")
            days = (end_date - start_date).days
            
            # Create evenly spaced dates
            dates = [start_date + timedelta(days=i * days / len(result.equity_curve)) 
                    for i in range(len(result.equity_curve))]
            
            # Create plotly figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, 
                y=result.equity_curve,
                mode='lines',
                name='Equity'
            ))
            
            # Add initial capital line
            fig.add_trace(go.Scatter(
                x=[dates[0], dates[-1]], 
                y=[result.initial_capital, result.initial_capital],
                mode='lines',
                name='Initial Capital',
                line=dict(dash='dash', color='gray')
            ))
            
            # Update layout
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Equity ($)",
                title="Equity Curve",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display trades
        if result.trades:
            st.subheader("Trades")
            
            # Create trades dataframe
            trades_df = pd.DataFrame(result.trades)
            
            # Limit to most relevant columns
            display_columns = ["symbol", "entry_date", "exit_date", "direction", 
                              "entry_price", "exit_price", "quantity", "pnl"]
            
            # Filter by available columns
            display_columns = [col for col in display_columns if col in trades_df.columns]
            
            if display_columns:
                st.dataframe(trades_df[display_columns], use_container_width=True)
        
        # Approval form
        st.subheader("Strategy Approval")
        
        with st.form("approval_form"):
            approval_decision = st.radio(
                "Approval Decision",
                ["Approve", "Reject"]
            )
            
            approver = st.text_input("Your Name", value="Admin")
            
            notes = st.text_area(
                "Notes",
                placeholder="Add any notes about this strategy..."
            )
            
            submitted = st.form_submit_button("Submit Decision")
            
            if submitted:
                approved = approval_decision == "Approve"
                
                # Update result
                result.set_approval(approved, approver, notes)
                
                # Save results
                self._save_backtest_results()
                
                if approved:
                    st.success(f"Strategy approved! It can now be deployed to paper trading.")
                else:
                    st.warning("Strategy rejected.")
                
                # Refresh the page
                st.experimental_rerun()
    
    def _render_approved_tab(self):
        """Render the approved strategies tab"""
        st.header("Approved Strategies")
        
        # Get approved results
        approved_results = {
            id: result for id, result in self.backtest_results.items()
            if result.approved
        }
        
        if not approved_results:
            st.info("No approved strategies yet.")
            return
        
        # Display approved strategies
        for result_id, result in approved_results.items():
            with st.expander(f"{result.strategy_name} - Approved on {result.approved_date.split('T')[0]}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**Return:** {result.returns_pct:.2f}%")
                    st.markdown(f"**Symbols:** {', '.join(result.symbols)}")
                    
                with col2:
                    st.markdown(f"**Sharpe Ratio:** {result.sharpe_ratio:.2f}")
                    st.markdown(f"**Win Rate:** {result.win_rate:.2f}%")
                
                with col3:
                    st.markdown(f"**Profit Factor:** {result.profit_factor:.2f}")
                    st.markdown(f"**Max Drawdown:** {result.max_drawdown:.2f}%")
                
                st.markdown(f"**Notes:** {result.notes}")
                
                # Deploy button
                if st.button("Deploy to Paper Trading", key=f"deploy_{result_id}"):
                    self._deploy_to_paper_trading(result_id, result)
                    st.success(f"Strategy deployed to paper trading!")
    
    def _run_backtest(self, strategy_name: str, symbols: List[str], start_date: str, 
                     end_date: str, initial_capital: float, params: Dict[str, Any]) -> Tuple[List[Dict], List[float]]:
        """
        Run a backtest of the specified strategy.
        
        For demonstration purposes, this generates simulated backtest results.
        In a real implementation, this would run the actual strategy on historical data.
        """
        # In a real implementation, we would:
        # 1. Fetch historical data for the specified symbols and date range
        # 2. Initialize the strategy with the provided parameters
        # 3. Run the strategy on the historical data
        # 4. Return the trades and equity curve
        
        # For demo purposes, generate simulated results based on strategy
        trades = []
        equity_curve = [initial_capital]
        
        # Number of days in backtest period
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end_dt - start_dt).days
        
        # Generate simulated trades
        num_trades = max(5, days // 5)  # Approximately one trade per week
        
        for i in range(num_trades):
            # Randomize trade parameters based on strategy characteristics
            if strategy_name == "Momentum":
                win_chance = 0.55
                avg_win = 0.03
                avg_loss = 0.02
            elif strategy_name == "Mean Reversion":
                win_chance = 0.65
                avg_win = 0.02
                avg_loss = 0.025
            elif strategy_name == "Trend Following":
                win_chance = 0.45
                avg_win = 0.06
                avg_loss = 0.03
            elif strategy_name == "Volatility Breakout":
                win_chance = 0.50
                avg_win = 0.04
                avg_loss = 0.03
            else:
                win_chance = 0.5
                avg_win = 0.03
                avg_loss = 0.03
            
            # Simulate trade outcome
            is_win = np.random.random() < win_chance
            symbol = np.random.choice(symbols)
            direction = np.random.choice(["LONG", "SHORT"])
            
            # Generate random dates within the backtest period
            entry_days = np.random.randint(0, days - 1)
            exit_days = entry_days + np.random.randint(1, 10)  # Hold between 1-10 days
            
            entry_date = (start_dt + timedelta(days=entry_days)).strftime("%Y-%m-%d")
            exit_date = (start_dt + timedelta(days=min(exit_days, days - 1))).strftime("%Y-%m-%d")
            
            # Calculate trade results
            base_price = 100 + (hash(symbol) % 200)  # Pseudo-random price
            entry_price = base_price * (1 + np.random.uniform(-0.05, 0.05))
            
            if is_win:
                pct_change = avg_win * (1 + np.random.uniform(-0.5, 0.5))
            else:
                pct_change = -avg_loss * (1 + np.random.uniform(-0.5, 0.5))
            
            if direction == "SHORT":
                pct_change = -pct_change
                
            exit_price = entry_price * (1 + pct_change)
            
            # Calculate position size (risk per trade)
            risk_pct = 0.01  # Risk 1% per trade
            position_value = equity_curve[-1] * risk_pct / avg_loss
            quantity = position_value / entry_price
            
            # Calculate P&L
            pnl = (exit_price - entry_price) * quantity
            if direction == "SHORT":
                pnl = -pnl
            
            # Add trade to list
            trades.append({
                "symbol": symbol,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "pnl": pnl
            })
            
            # Update equity curve
            equity_curve.append(equity_curve[-1] + pnl)
        
        # Sort trades by entry date
        trades.sort(key=lambda x: x["entry_date"])
        
        # Generate more granular equity curve
        detailed_equity = [initial_capital]
        days_per_point = max(1, days // 100)
        
        current_equity = initial_capital
        next_trade_idx = 0
        
        for day in range(days):
            current_date = (start_dt + timedelta(days=day)).strftime("%Y-%m-%d")
            
            # Apply any completed trades
            while next_trade_idx < len(trades) and trades[next_trade_idx]["exit_date"] <= current_date:
                current_equity += trades[next_trade_idx]["pnl"]
                next_trade_idx += 1
            
            # Add equity point at specified intervals
            if day % days_per_point == 0:
                detailed_equity.append(current_equity)
        
        # Ensure the final equity value is included
        if detailed_equity[-1] != current_equity:
            detailed_equity.append(current_equity)
        
        return trades, detailed_equity
    
    def _deploy_to_paper_trading(self, result_id: str, result: BacktestResult):
        """
        Deploy an approved strategy to paper trading.
        
        In a real implementation, this would:
        1. Initialize the strategy with the approved parameters
        2. Connect to the paper trading API
        3. Set up the strategy to run in paper trading mode
        """
        # Create the deployment directory if it doesn't exist
        deploy_dir = os.path.join(
            os.path.dirname(self.data_dir),
            "paper_trading"
        )
        os.makedirs(deploy_dir, exist_ok=True)
        
        # Create deployment configuration
        deployment = {
            "result_id": result_id,
            "strategy_name": result.strategy_name,
            "symbols": result.symbols,
            "params": result.params,
            "initial_capital": result.initial_capital,
            "deployed_date": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Save deployment configuration
        deploy_file = os.path.join(deploy_dir, f"{result_id}.json")
        with open(deploy_file, 'w') as f:
            json.dump(deployment, f, indent=2)
    
    def _load_backtest_results(self) -> Dict[str, BacktestResult]:
        """Load backtest results from disk"""
        results = {}
        
        # Look for result files
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                try:
                    file_path = os.path.join(self.data_dir, filename)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    # Create result object
                    result_id = os.path.splitext(filename)[0]
                    results[result_id] = BacktestResult.from_dict(data)
                except Exception as e:
                    logger.error(f"Error loading backtest result {filename}: {e}")
        
        return results
    
    def _save_backtest_results(self):
        """Save backtest results to disk"""
        for result_id, result in self.backtest_results.items():
            try:
                file_path = os.path.join(self.data_dir, f"{result_id}.json")
                with open(file_path, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
            except Exception as e:
                logger.error(f"Error saving backtest result {result_id}: {e}")
