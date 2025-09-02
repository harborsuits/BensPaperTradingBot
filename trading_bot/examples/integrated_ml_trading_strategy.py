#!/usr/bin/env python3
"""
Integrated ML Trading Strategy Example

This script demonstrates how to combine multiple ML components including:
- Market condition classification
- Price prediction models
- Anomaly detection
- Parameter optimization

Together, these components create a robust trading strategy that:
1. Adapts to changing market regimes
2. Identifies microstructure anomalies for risk management
3. Adjusts position sizing based on model confidence and anomaly detection
4. Makes trading decisions with layered ML signals

Usage:
    python integrated_ml_trading_strategy.py --symbol SPY --backtest
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

# Add trading_bot to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Rich console outputs
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

# Import trading components
from multi_asset_adapter import MultiAssetAdapter
from risk_manager import RiskManager, RiskLevel
from ml.market_anomaly_detector import MarketAnomalyDetector
from ml.price_prediction_model import PricePredictionModel
from ml.market_condition_classifier import MarketConditionClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize console for rich output
console = Console()

class IntegratedMLStrategy:
    """
    Integrated ML trading strategy that combines multiple ML components.
    
    This strategy integrates:
    - Market condition classifiers to identify market regimes
    - Price prediction models for directional bias
    - Anomaly detection for risk management
    - Parameter optimization based on market conditions
    """
    
    def __init__(self, 
                 symbol: str,
                 models_dir: str = "models",
                 adapter: Optional[MultiAssetAdapter] = None,
                 risk_manager: Optional[RiskManager] = None,
                 timeframe: str = "1h"):
        """
        Initialize the integrated ML strategy.
        
        Args:
            symbol: Trading instrument symbol
            models_dir: Directory to store trained models
            adapter: Optional MultiAssetAdapter instance
            risk_manager: Optional RiskManager instance
            timeframe: Default timeframe for data
        """
        self.symbol = symbol
        self.models_dir = models_dir
        self.adapter = adapter
        self.risk_manager = risk_manager
        self.timeframe = timeframe
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize ML components
        self._initialize_ml_components()
        
        # Trading parameters
        self.strategy_params = {
            'position_size': 1.0,  # Base position size
            'risk_per_trade': 0.01,  # 1% risk per trade
            'stop_loss_atr_mult': 2.0,  # Stop loss ATR multiplier
            'take_profit_atr_mult': 3.0,  # Take profit ATR multiplier
            'max_holding_period': 10,  # Maximum holding period in bars
            'anomaly_position_reduction': 0.5,  # Reduce position by 50% on anomaly
            'confidence_threshold': 0.65,  # Minimum confidence for trading
        }
        
        # Strategy state
        self.current_market_condition = None
        self.current_position = 0  # -1 (short), 0 (flat), 1 (long)
        self.entry_price = None
        self.entry_time = None
        self.stop_loss = None
        self.take_profit = None
        self.last_prediction = None
        self.last_anomaly_status = None
        
        logger.info(f"Initialized integrated ML strategy for {symbol}")
    
    def _initialize_ml_components(self):
        """Initialize the ML components."""
        # Initialize market condition classifier
        self.condition_classifier = MarketConditionClassifier(
            symbol=self.symbol,
            model_dir=os.path.join(self.models_dir, "market_condition"),
            model_type="random_forest"
        )
        
        # Initialize price prediction model
        self.price_predictor = PricePredictionModel(
            prediction_horizon=5,
            confidence_threshold=0.65,
            model_type='gradient_boosting',
            model_path=os.path.join(self.models_dir, f"price_pred_{self.symbol}")
        )
        
        # Initialize anomaly detector
        self.anomaly_detector = MarketAnomalyDetector(
            symbol=self.symbol,
            lookback_window=20,
            alert_threshold=0.75,
            model_dir=os.path.join(self.models_dir, "anomaly_detection"),
            use_autoencoder=True
        )
        
        # Try to load existing models
        try:
            self.price_predictor.load_model()
            logger.info(f"Loaded existing price prediction model for {self.symbol}")
        except:
            logger.info(f"No existing price prediction model found for {self.symbol}")
        
        try:
            self.anomaly_detector.load_models()
            logger.info(f"Loaded existing anomaly detection model for {self.symbol}")
        except:
            logger.info(f"No existing anomaly detection model found for {self.symbol}")
            
        # Parameters for different market conditions
        self.condition_params = {
            "bullish_trend": {
                'position_size': 1.0,
                'risk_per_trade': 0.01,
                'stop_loss_atr_mult': 2.0,
                'take_profit_atr_mult': 4.0,
            },
            "bearish_trend": {
                'position_size': 0.75,
                'risk_per_trade': 0.008,
                'stop_loss_atr_mult': 1.5,
                'take_profit_atr_mult': 3.0,
            },
            "sideways": {
                'position_size': 0.5,
                'risk_per_trade': 0.005,
                'stop_loss_atr_mult': 1.0,
                'take_profit_atr_mult': 2.0,
            },
            "high_volatility": {
                'position_size': 0.25,
                'risk_per_trade': 0.005,
                'stop_loss_atr_mult': 3.0,
                'take_profit_atr_mult': 5.0,
            }
        }
    
    def train_models(self, data: pd.DataFrame, force_training: bool = False) -> Dict[str, Any]:
        """
        Train all ML models.
        
        Args:
            data: DataFrame with historical OHLCV data
            force_training: Force retraining even if models exist
            
        Returns:
            Dict with training results
        """
        console.print(Panel("Training ML models", style="bold blue"))
        
        results = {
            "market_condition": False,
            "price_prediction": False,
            "anomaly_detection": False
        }
        
        with Progress() as progress:
            # Set up progress tracking
            task1 = progress.add_task("Training market condition classifier...", total=100)
            task2 = progress.add_task("Training price predictor...", total=100)
            task3 = progress.add_task("Training anomaly detector...", total=100)
            
            # Train market condition classifier
            progress.update(task1, advance=30)
            try:
                self.condition_classifier.train(data)
                results["market_condition"] = True
                progress.update(task1, completed=100)
            except Exception as e:
                logger.error(f"Error training market condition classifier: {e}")
                progress.update(task1, completed=100, description=f"[red]Error: {e}[/red]")
            
            # Train price predictor
            progress.update(task2, advance=30)
            try:
                self.price_predictor.train(data, model_key=self.symbol)
                self.price_predictor.save_model()
                results["price_prediction"] = True
                progress.update(task2, completed=100)
            except Exception as e:
                logger.error(f"Error training price predictor: {e}")
                progress.update(task2, completed=100, description=f"[red]Error: {e}[/red]")
            
            # Train anomaly detector
            progress.update(task3, advance=30)
            try:
                self.anomaly_detector.train(data)
                self.anomaly_detector.save_model()
                results["anomaly_detection"] = True
                progress.update(task3, completed=100)
            except Exception as e:
                logger.error(f"Error training anomaly detector: {e}")
                progress.update(task3, completed=100, description=f"[red]Error: {e}[/red]")
        
        return results
    
    def analyze_market_condition(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the current market condition.
        
        Args:
            data: DataFrame with recent OHLCV data
            
        Returns:
            Dict with market condition analysis results
        """
        try:
            result = self.condition_classifier.predict(data)
            self.current_market_condition = result.get("market_condition")
            
            # Update strategy parameters based on market condition
            if self.current_market_condition and self.current_market_condition in self.condition_params:
                for param, value in self.condition_params[self.current_market_condition].items():
                    self.strategy_params[param] = value
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing market condition: {e}")
            return {"error": str(e)}
    
    def predict_price_movement(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict future price movement.
        
        Args:
            data: DataFrame with recent OHLCV data
            
        Returns:
            Dict with price prediction results
        """
        try:
            prediction = self.price_predictor.predict(data, model_key=self.symbol)
            self.last_prediction = prediction
            return prediction
        except Exception as e:
            logger.error(f"Error predicting price movement: {e}")
            return {"error": str(e)}
    
    def detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect market microstructure anomalies.
        
        Args:
            data: DataFrame with recent OHLCV data
            
        Returns:
            Dict with anomaly detection results
        """
        try:
            result = self.anomaly_detector.detect_anomalies(data)
            
            # Store anomaly status for decision making
            is_anomaly = False
            if "latest_score" in result and result["latest_score"] > self.anomaly_detector.alert_threshold:
                is_anomaly = True
            
            self.last_anomaly_status = {
                "is_anomaly": is_anomaly,
                "score": result.get("latest_score", 0),
                "details": result
            }
            
            return result
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {"error": str(e)}
    
    def generate_trading_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive trading signal.
        
        This combines market condition, price prediction, and anomaly detection
        to produce a final trading decision.
        
        Args:
            data: DataFrame with recent OHLCV data
            
        Returns:
            Dict with trading signal details
        """
        # Run all ML components to get latest insights
        market_condition = self.analyze_market_condition(data)
        price_prediction = self.predict_price_movement(data)
        anomaly_result = self.detect_anomalies(data)
        
        # Initialize default signal
        signal = {
            "action": "HOLD",
            "position_size": 0,
            "confidence": 0,
            "stop_loss": None,
            "take_profit": None,
            "reasoning": []
        }
        
        # Check for errors in any component
        if "error" in market_condition or "error" in price_prediction or "error" in anomaly_result:
            signal["reasoning"].append("Error in one or more ML components")
            return signal
        
        # Check if anomaly is detected
        is_anomaly = self.last_anomaly_status and self.last_anomaly_status["is_anomaly"]
        if is_anomaly:
            signal["reasoning"].append(f"Market anomaly detected (score: {self.last_anomaly_status['score']:.4f})")
            
            # If we have an open position, recommend reducing size
            if self.current_position != 0:
                signal["action"] = "REDUCE" if self.current_position > 0 else "REDUCE_SHORT"
                signal["position_size"] = self.strategy_params["anomaly_position_reduction"]
                signal["reasoning"].append(f"Reducing position due to anomaly")
                return signal
        
        # Get prediction details
        if "direction" in price_prediction:
            direction = price_prediction["direction"]
            confidence = price_prediction.get("confidence", 0)
            
            # Only generate a signal if confidence is above threshold
            if confidence >= self.strategy_params["confidence_threshold"]:
                if direction == "up":
                    signal["action"] = "BUY"
                    signal["reasoning"].append(f"Bullish prediction with {confidence:.2f} confidence")
                elif direction == "down":
                    signal["action"] = "SELL"
                    signal["reasoning"].append(f"Bearish prediction with {confidence:.2f} confidence")
                
                signal["confidence"] = confidence
                
                # Adjust position size based on market condition and confidence
                base_size = self.strategy_params["position_size"]
                confidence_factor = confidence / self.strategy_params["confidence_threshold"]
                
                # Reduce position if anomaly was detected
                anomaly_factor = self.strategy_params["anomaly_position_reduction"] if is_anomaly else 1.0
                
                signal["position_size"] = base_size * confidence_factor * anomaly_factor
                
                # Calculate stop loss and take profit based on ATR if available
                try:
                    # Get current ATR
                    atr = data['atr'].iloc[-1] if 'atr' in data.columns else self._calculate_atr(data)
                    
                    current_price = data['close'].iloc[-1]
                    
                    if direction == "up":
                        signal["stop_loss"] = current_price - (atr * self.strategy_params["stop_loss_atr_mult"])
                        signal["take_profit"] = current_price + (atr * self.strategy_params["take_profit_atr_mult"])
                    else:
                        signal["stop_loss"] = current_price + (atr * self.strategy_params["stop_loss_atr_mult"])
                        signal["take_profit"] = current_price - (atr * self.strategy_params["take_profit_atr_mult"])
                except:
                    # If ATR calculation fails, don't set stop loss and take profit
                    pass
            else:
                signal["reasoning"].append(f"Prediction confidence ({confidence:.2f}) below threshold ({self.strategy_params['confidence_threshold']:.2f})")
        
        # Add market condition context
        if self.current_market_condition:
            signal["reasoning"].append(f"Current market condition: {self.current_market_condition}")
            signal["market_condition"] = self.current_market_condition
        
        return signal
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            high = data['high']
            low = data['low']
            close = data['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            
            return atr
        except:
            # Default to 2% of price if calculation fails
            return data['close'].iloc[-1] * 0.02
    
    def execute_signal(self, signal: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal dictionary
            data: Current market data
            
        Returns:
            Dict with execution results
        """
        result = {
            "executed": False,
            "action": signal["action"],
            "price": data['close'].iloc[-1],
            "time": data.index[-1],
            "position_size": signal["position_size"],
            "reason": ", ".join(signal["reasoning"])
        }
        
        # Update position tracking
        if signal["action"] == "BUY" and self.current_position <= 0:
            # Enter long position
            self.current_position = 1
            self.entry_price = result["price"]
            self.entry_time = result["time"]
            self.stop_loss = signal["stop_loss"]
            self.take_profit = signal["take_profit"]
            result["executed"] = True
            
        elif signal["action"] == "SELL" and self.current_position >= 0:
            # Enter short position
            self.current_position = -1
            self.entry_price = result["price"]
            self.entry_time = result["time"]
            self.stop_loss = signal["stop_loss"]
            self.take_profit = signal["take_profit"]
            result["executed"] = True
            
        elif signal["action"] == "REDUCE" and self.current_position > 0:
            # Reduce long position
            self.current_position = signal["position_size"]
            result["executed"] = True
            result["partial"] = True
            
        elif signal["action"] == "REDUCE_SHORT" and self.current_position < 0:
            # Reduce short position
            self.current_position = -signal["position_size"]
            result["executed"] = True
            result["partial"] = True
            
        elif signal["action"] == "CLOSE" or signal["action"] == "EXIT":
            # Close position
            self.current_position = 0
            self.entry_price = None
            self.entry_time = None
            self.stop_loss = None
            self.take_profit = None
            result["executed"] = True
        
        if result["executed"]:
            logger.info(f"Executed {signal['action']} at {result['price']}, size={result['position_size']}")
        
        return result
    
    def check_exits(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check if any exit conditions are met.
        
        Args:
            data: Current market data
            
        Returns:
            Dict with exit signal if needed
        """
        if self.current_position == 0 or self.entry_price is None:
            return None
        
        current_price = data['close'].iloc[-1]
        current_time = data.index[-1]
        
        # Initialize exit signal
        exit_signal = None
        
        # Check stop loss
        if self.stop_loss is not None:
            if (self.current_position > 0 and current_price <= self.stop_loss) or \
               (self.current_position < 0 and current_price >= self.stop_loss):
                exit_signal = {
                    "action": "CLOSE",
                    "position_size": 0,
                    "confidence": 1.0,
                    "reasoning": ["Stop loss triggered"]
                }
        
        # Check take profit
        if self.take_profit is not None and exit_signal is None:
            if (self.current_position > 0 and current_price >= self.take_profit) or \
               (self.current_position < 0 and current_price <= self.take_profit):
                exit_signal = {
                    "action": "CLOSE",
                    "position_size": 0,
                    "confidence": 1.0,
                    "reasoning": ["Take profit triggered"]
                }
        
        # Check maximum holding period
        if exit_signal is None and self.entry_time is not None:
            holding_period = (current_time - self.entry_time).total_seconds() / 3600
            if holding_period > self.strategy_params["max_holding_period"]:
                exit_signal = {
                    "action": "CLOSE",
                    "position_size": 0,
                    "confidence": 1.0,
                    "reasoning": ["Maximum holding period reached"]
                }
        
        # Check for anomalies as exit trigger
        if exit_signal is None and self.last_anomaly_status and self.last_anomaly_status["is_anomaly"]:
            # If anomaly is severe enough, exit position
            if self.last_anomaly_status["score"] > 0.9:  # High severity anomaly
                exit_signal = {
                    "action": "CLOSE",
                    "position_size": 0,
                    "confidence": 1.0,
                    "reasoning": ["Severe market anomaly detected"]
                }
        
        return exit_signal
    
    def run_backtest(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run a backtest of the integrated ML strategy.
        
        Args:
            historical_data: DataFrame with historical OHLCV data
            
        Returns:
            Dict with backtest results
        """
        console.print(Panel("Running backtest with integrated ML strategy", style="bold blue"))
        
        # Make a copy of the historical data
        data = historical_data.copy()
        
        # Minimum required data length
        min_data_length = 100
        if len(data) < min_data_length:
            return {"error": f"Insufficient data for backtesting. Need at least {min_data_length} bars."}
        
        # Train models if needed (using the first 70% of data)
        train_idx = int(len(data) * 0.7)
        training_data = data.iloc[:train_idx]
        
        with console.status("[bold green]Training models for backtest..."):
            self.train_models(training_data)
        
        # Initialize backtest results
        results = {
            "trades": [],
            "equity_curve": [1.0],  # Start with $1
            "positions": [0],
            "timestamps": [data.index[min_data_length-1]],
            "signals": []
        }
        
        # Initial capital
        equity = 1.0
        position = 0
        
        # Loop through the test data
        console.print("[bold green]Running backtest simulation...[/bold green]")
        
        with Progress() as progress:
            bar = progress.add_task("[cyan]Backtesting...", total=len(data)-min_data_length)
            
            for i in range(min_data_length, len(data)):
                # Get window of data for this bar
                window = data.iloc[i-min_data_length:i+1]
                
                # Check for exits first
                exit_signal = self.check_exits(window)
                if exit_signal is not None:
                    # Execute exit
                    execution = self.execute_signal(exit_signal, window)
                    if execution["executed"]:
                        # Record the trade result
                        exit_price = window['close'].iloc[-1]
                        pnl = 0
                        
                        if position > 0:  # Long position
                            pnl = (exit_price / self.entry_price - 1) * position
                        elif position < 0:  # Short position
                            pnl = (self.entry_price / exit_price - 1) * abs(position)
                        
                        trade = {
                            "entry_time": self.entry_time,
                            "entry_price": self.entry_price,
                            "exit_time": window.index[-1],
                            "exit_price": exit_price,
                            "position": position,
                            "pnl": pnl,
                            "reason": ", ".join(exit_signal["reasoning"])
                        }
                        
                        results["trades"].append(trade)
                        
                        # Update equity
                        equity *= (1 + pnl)
                        position = 0
                
                # Generate trading signal
                signal = self.generate_trading_signal(window)
                results["signals"].append({
                    "time": window.index[-1],
                    "signal": signal["action"],
                    "confidence": signal.get("confidence", 0),
                    "reasoning": signal["reasoning"]
                })
                
                # Execute signal if not already in a position matching the signal
                if (signal["action"] == "BUY" and position <= 0) or \
                   (signal["action"] == "SELL" and position >= 0):
                    execution = self.execute_signal(signal, window)
                    
                    if execution["executed"]:
                        # Update position
                        position = self.current_position
                        results["trades"].append({
                            "entry_time": window.index[-1],
                            "entry_price": window['close'].iloc[-1],
                            "position": position,
                            "stop_loss": signal.get("stop_loss"),
                            "take_profit": signal.get("take_profit"),
                            "reason": ", ".join(signal["reasoning"])
                        })
                
                # Update equity curve with current position value
                price_change = 0
                if i > min_data_length:
                    prev_close = data['close'].iloc[i-1]
                    curr_close = data['close'].iloc[i]
                    price_change = curr_close / prev_close - 1
                
                if position != 0:
                    equity_change = position * price_change  # simple return calculation
                    equity *= (1 + equity_change)
                
                # Record equity and position
                results["equity_curve"].append(equity)
                results["positions"].append(position)
                results["timestamps"].append(window.index[-1])
                
                # Update progress bar
                progress.update(bar, advance=1)
        
        # Calculate performance metrics
        equity_curve = np.array(results["equity_curve"])
        total_return = equity_curve[-1] / equity_curve[0] - 1
        
        # Get all trades with exit
        completed_trades = [t for t in results["trades"] if "exit_price" in t]
        win_trades = [t for t in completed_trades if t["pnl"] > 0]
        
        win_rate = len(win_trades) / len(completed_trades) if completed_trades else 0
        
        # Calculate drawdowns
        drawdowns = 1 - equity_curve / np.maximum.accumulate(equity_curve)
        max_drawdown = np.max(drawdowns)
        
        # Calculate daily returns and volatility
        if isinstance(results["timestamps"][0], pd.Timestamp):
            daily_returns = pd.Series(np.diff(np.log(equity_curve)), index=results["timestamps"][1:]).resample('D').sum()
            annualized_return = np.exp(daily_returns.mean() * 252) - 1
            annualized_vol = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        else:
            # Simplified calculation if timestamps are not datetime objects
            returns = np.diff(np.log(equity_curve))
            annualized_return = np.mean(returns) * 252
            annualized_vol = np.std(returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Add metrics to results
        results["metrics"] = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": len(completed_trades),
            "win_trades": len(win_trades),
            "loss_trades": len(completed_trades) - len(win_trades)
        }
        
        console.print("[bold green]Backtest completed successfully![/bold green]")
        return results

def display_backtest_results(results):
    """Display the backtest results."""
    if "error" in results:
        console.print(f"[bold red]Error: {results['error']}[/bold red]")
        return
    
    metrics = results["metrics"]
    trades = results["trades"]
    completed_trades = [t for t in trades if "exit_price" in t]
    
    # Create performance table
    table = Table(title="Backtest Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Return", f"{metrics['total_return']*100:.2f}%")
    table.add_row("Annualized Return", f"{metrics['annualized_return']*100:.2f}%")
    table.add_row("Annualized Volatility", f"{metrics['annualized_volatility']*100:.2f}%")
    table.add_row("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    table.add_row("Maximum Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
    table.add_row("Win Rate", f"{metrics['win_rate']*100:.2f}%")
    table.add_row("Total Trades", str(metrics['total_trades']))
    table.add_row("Winning Trades", str(metrics['win_trades']))
    table.add_row("Losing Trades", str(metrics['loss_trades']))
    
    console.print(table)
    
    # Display last few trades
    if completed_trades:
        console.print("\n[bold]Recent Trades:[/bold]")
        for trade in completed_trades[-5:]:
            position_type = "LONG" if trade["position"] > 0 else "SHORT"
            pnl = trade.get("pnl", 0)
            pnl_str = f"{pnl*100:.2f}%" if "pnl" in trade else "Open"
            result = "[green]WIN[/green]" if pnl > 0 else "[red]LOSS[/red]" if pnl < 0 else "N/A"
            
            console.print(f"{position_type} {trade['entry_time']} @ {trade['entry_price']:.2f} → ", end="")
            if "exit_time" in trade:
                console.print(f"{trade['exit_time']} @ {trade['exit_price']:.2f} | {pnl_str} | {result}")
            else:
                console.print("Open position")

def plot_backtest_results(results, title="Integrated ML Strategy Backtest"):
    """Plot the backtest results."""
    if "error" in results:
        return
    
    equity_curve = results["equity_curve"]
    positions = results["positions"]
    timestamps = results["timestamps"]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    ax1.plot(timestamps, equity_curve, label="Portfolio Value", color="blue")
    ax1.set_title(title)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.grid(True)
    
    # Add markers for trades
    for trade in results["trades"]:
        if "exit_time" in trade and "entry_time" in trade:
            # Completed trade
            entry_time = trade["entry_time"]
            exit_time = trade["exit_time"]
            entry_value = equity_curve[timestamps.index(entry_time)]
            exit_value = equity_curve[timestamps.index(exit_time)]
            
            # Color based on profit/loss
            color = "green" if trade.get("pnl", 0) > 0 else "red"
            ax1.plot([entry_time, exit_time], [entry_value, exit_value], color=color, linewidth=1.5, alpha=0.7)
    
    # Plot drawdown
    equity_array = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array / running_max - 1) * 100  # Convert to percentage
    
    ax2.fill_between(timestamps, 0, drawdown, color="red", alpha=0.3, label="Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save figure to output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filename)
    console.print(f"[green]Plot saved to {filename}[/green]")
    
    plt.show()

def fetch_data(symbol, adapter=None, days=180, timeframe="1h"):
    """Fetch historical data for the symbol."""
    try:
        if adapter:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = adapter.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            
            if data is not None and len(data) > 0:
                console.print(f"[green]Successfully loaded {len(data)} bars for {symbol}[/green]")
                return data
        
        # Fall back to mock data if adapter fails or is not available
        return generate_mock_data(symbol, days, timeframe)
    except Exception as e:
        console.print(f"[yellow]Error fetching data: {str(e)}. Using mock data.[/yellow]")
        return generate_mock_data(symbol, days, timeframe)

def generate_mock_data(symbol, days, timeframe):
    """Generate mock data for testing."""
    console.print(f"[yellow]Generating mock data for {symbol}...[/yellow]")
    
    # Determine number of bars based on timeframe
    bars_per_day = 24
    if timeframe.endswith("h"):
        hours = int(timeframe[:-1])
        bars_per_day = 24 // hours
    elif timeframe.endswith("d"):
        bars_per_day = 1
    
    total_bars = days * bars_per_day
    
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, periods=total_bars)
    
    # Generate price data with trend and noise
    np.random.seed(42)  # For reproducibility
    
    # Start price
    price = 100.0
    if "BTC" in symbol:
        price = 30000.0
    elif "ETH" in symbol:
        price = 2000.0
    
    # Generate price series with random walk, trend, and cycles
    returns = np.random.normal(0.0001, 0.001, total_bars)  # Random noise
    
    # Add a trend component
    trend = np.linspace(0, 0.2, total_bars)
    if np.random.random() < 0.5:  # 50% chance of downtrend
        trend = -trend
    
    # Add cyclical component
    cycle = 0.1 * np.sin(np.linspace(0, 10, total_bars))
    
    # Combine components
    cumulative_returns = np.cumsum(returns) + trend + cycle
    prices = price * (1 + cumulative_returns)
    
    # Generate OHLCV data
    volatility = 0.002  # Daily volatility
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, volatility, total_bars)),
        'high': prices * (1 + np.abs(np.random.normal(0, volatility * 2, total_bars))),
        'low': prices * (1 - np.abs(np.random.normal(0, volatility * 2, total_bars))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, total_bars)
    }, index=dates)
    
    # Add some technical indicators
    data['atr'] = _calculate_mock_atr(data)
    
    console.print(f"[green]Generated {len(data)} bars of mock data[/green]")
    return data

def _calculate_mock_atr(data, period=14):
    """Calculate ATR for mock data."""
    high = data['high']
    low = data['low']
    close = data['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return atr

def main():
    """Main function to run the integrated ML trading strategy example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Integrated ML Trading Strategy Example")
    parser.add_argument("--symbol", default="BTC/USD", help="Trading symbol to analyze")
    parser.add_argument("--timeframe", default="1h", help="Timeframe for analysis")
    parser.add_argument("--days", type=int, default=180, help="Days of historical data")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    args = parser.parse_args()
    
    console.print(Panel(f"Integrated ML Trading Strategy", 
                        subtitle=f"Symbol: {args.symbol}, Timeframe: {args.timeframe}",
                        style="bold blue"))
    
    # Initialize components
    try:
        # Try to initialize adapter
        adapter = MultiAssetAdapter()
        console.print("[green]MultiAssetAdapter initialized successfully[/green]")
    except:
        console.print("[yellow]Could not initialize MultiAssetAdapter, using mock data[/yellow]")
        adapter = None
    
    # Fetch historical data
    historical_data = fetch_data(args.symbol, adapter, args.days, args.timeframe)
    
    if historical_data is None or len(historical_data) < 100:
        console.print("[red]Insufficient data for analysis. Exiting.[/red]")
        return
    
    # Initialize integrated ML strategy
    strategy = IntegratedMLStrategy(
        symbol=args.symbol,
        adapter=adapter,
        timeframe=args.timeframe
    )
    
    if args.backtest:
        # Run backtest
        backtest_results = strategy.run_backtest(historical_data)
        
        # Display backtest results
        display_backtest_results(backtest_results)
        
        # Plot results if requested
        if args.plot:
            plot_backtest_results(backtest_results, f"{args.symbol} - Integrated ML Strategy")
    else:
        # Just analyze latest data
        console.print("[bold]Analyzing latest market data...[/bold]")
        
        # Get the last 100 bars for analysis
        latest_data = historical_data.iloc[-100:].copy()
        
        # Train models
        strategy.train_models(historical_data)
        
        # Generate trading signal
        signal = strategy.generate_trading_signal(latest_data)
        
        # Display the analysis results
        console.print(Panel(f"Trading Signal: [bold]{signal['action']}[/bold]", style="cyan"))
        console.print(f"Confidence: {signal.get('confidence', 0):.2f}")
        console.print(f"Position Size: {signal.get('position_size', 0):.2f}")
        
        if signal.get("stop_loss"):
            console.print(f"Stop Loss: {signal['stop_loss']:.2f}")
        
        if signal.get("take_profit"):
            console.print(f"Take Profit: {signal['take_profit']:.2f}")
        
        console.print("\n[bold]Reasoning:[/bold]")
        for reason in signal["reasoning"]:
            console.print(f"• {reason}")
    
    console.print("\n[bold green]Analysis complete![/bold green]")

if __name__ == "__main__":
    main() 