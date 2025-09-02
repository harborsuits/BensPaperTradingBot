#!/usr/bin/env python3
"""
Anomaly Risk Response Example

This script demonstrates how the trading system responds to detected market anomalies
by adjusting risk parameters, position sizing, and trade execution.
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from multi_asset_adapter import MultiAssetAdapter
from risk_manager import RiskManager
from ml.market_anomaly_detector import MarketAnomalyDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize console for rich output
console = Console()

class MockTradeExecutor:
    """Mock trade executor for demonstration purposes."""
    
    def __init__(self, adapter, risk_manager):
        """Initialize the mock trade executor."""
        self.adapter = adapter
        self.risk_manager = risk_manager
        self.trades = []
        
    def submit_trade(self, symbol, direction, quantity, entry_price, stop_price, target_price=None):
        """Submit a trade for execution."""
        # First, check with risk manager
        risk_assessment = self.risk_manager.check_trade_risk(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price
        )
        
        # Display the risk assessment
        self._display_risk_assessment(risk_assessment)
        
        # Check if trade passes risk checks
        if risk_assessment["risk_checks_passed"]:
            # Check for anomaly cooldown
            is_cooldown, cooldown_end = self.risk_manager.is_in_anomaly_cooldown()
            if is_cooldown:
                console.print("[yellow]Note: Executing during anomaly cooldown period[/yellow]")
                
                # Get risk status
                risk_status = self.risk_manager.get_anomaly_risk_status()
                
                # Check if trading is restricted
                if risk_status.get("trading_restricted", False):
                    console.print("[bold red]Trading restricted due to anomaly - trade rejected[/bold red]")
                    return None
            
            # Execute the trade
            trade_id = f"T{len(self.trades) + 1}"
            trade = {
                "id": trade_id,
                "symbol": symbol,
                "direction": direction,
                "quantity": risk_assessment.get("quantity", quantity),
                "entry_price": entry_price,
                "stop_price": risk_assessment.get("stop_price", stop_price),
                "target_price": target_price,
                "status": "executed",
                "time": datetime.now(),
                "risk_assessment": risk_assessment
            }
            
            self.trades.append(trade)
            
            console.print(f"[green]Trade {trade_id} executed successfully[/green]")
            return trade
        else:
            console.print("[bold red]Trade rejected due to risk assessment[/bold red]")
            return None
    
    def _display_risk_assessment(self, assessment):
        """Display the risk assessment results."""
        # Create a table for risk details
        table = Table(title="Risk Assessment")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="yellow")
        
        # Add basic details
        table.add_row("Symbol", assessment["symbol"])
        table.add_row("Direction", assessment["direction"])
        table.add_row("Quantity", str(assessment["quantity"]))
        table.add_row("Entry Price", f"{assessment['entry_price']:.2f}")
        table.add_row("Stop Price", f"{assessment['stop_price']:.2f}")
        
        if assessment["target_price"]:
            table.add_row("Target Price", f"{assessment['target_price']:.2f}")
        
        # Add risk details
        if "risk_details" in assessment:
            details = assessment["risk_details"]
            for key, value in details.items():
                if key == "risk_percentage":
                    table.add_row("Risk %", f"{value:.2f}%")
                elif key == "dollar_risk":
                    table.add_row("Dollar Risk", f"${value:.2f}")
                elif key == "reward_risk_ratio" and value is not None:
                    table.add_row("Reward/Risk", f"{value:.2f}")
                elif key != "account_size":
                    table.add_row(key, str(value))
        
        # Add anomaly adjustments if present
        if "adjusted_for_anomaly" in assessment and assessment["adjusted_for_anomaly"]:
            table.add_row("Anomaly Adjusted", "Yes")
            table.add_row("Position Size Modifier", f"{assessment['position_size_modifier']:.2f}")
        
        console.print(table)
        
        # Display warnings and errors
        if assessment["warnings"]:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in assessment["warnings"]:
                console.print(f"• {warning}")
        
        if assessment["errors"]:
            console.print("\n[bold red]Errors:[/bold red]")
            for error in assessment["errors"]:
                console.print(f"• {error}")

def generate_mock_data(symbol, periods=100):
    """Generate mock market data with some anomalies."""
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=periods)
    date_range = pd.date_range(start=start_date, end=end_date, periods=periods)
    
    # Generate price data
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.001, periods).cumsum()
    price_series = base_price * (1 + returns)
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': price_series * (1 + 0.001 * np.random.randn(periods)),
        'high': price_series * (1 + 0.003 + 0.002 * np.random.rand(periods)),
        'low': price_series * (1 - 0.003 - 0.002 * np.random.rand(periods)),
        'close': price_series,
        'volume': np.random.lognormal(10, 1, periods),
    }, index=date_range)
    
    # Add bid-ask data
    data['bid'] = data['close'] * 0.999
    data['ask'] = data['close'] * 1.001
    
    # Add order book data
    data['bid_size'] = np.random.lognormal(8, 1, periods)
    data['ask_size'] = np.random.lognormal(8, 1, periods)
    
    # Add anomalies at specific points for demonstration
    
    # 1. Price spike near the end
    anomaly_idx1 = periods - 5
    data.iloc[anomaly_idx1, data.columns.get_indexer(['high', 'close'])] *= 1.05
    
    # 2. Volume spike in the middle
    anomaly_idx2 = periods // 2
    data.iloc[anomaly_idx2, data.columns.get_indexer(['volume'])] *= 10
    
    # 3. Bid-ask spread widening
    anomaly_idx3 = periods - 10
    data.iloc[anomaly_idx3, data.columns.get_indexer(['bid'])] *= 0.97
    data.iloc[anomaly_idx3, data.columns.get_indexer(['ask'])] *= 1.03
    
    console.print(f"[green]Generated mock data for {symbol} with {periods} periods and 3 injected anomalies[/green]")
    return data

def simulate_trading_with_anomalies():
    """Simulate trading with anomaly detection and risk management."""
    console.print(Panel("Anomaly Risk Response Simulation", style="bold blue"))
    
    # Initialize the trading components
    symbol = "AAPL"
    
    # 1. Create a mock adapter
    adapter = MultiAssetAdapter({})
    
    # Override get_account_balance for demo purposes
    adapter.get_account_balance = lambda: 100000.0
    
    # 2. Create risk manager
    risk_manager = RiskManager(
        multi_asset_adapter=adapter,
        journal_dir="journal",
        anomaly_config_path="trading_bot/config/anomaly_risk_rules.yaml"
    )
    
    # 3. Create anomaly detector
    detector = MarketAnomalyDetector(
        symbol=symbol,
        lookback_window=10,
        alert_threshold=0.75,
        model_dir="models/anomaly_detection",
        use_autoencoder=True
    )
    
    # 4. Create mock trade executor
    executor = MockTradeExecutor(adapter, risk_manager)
    
    # 5. Generate mock data
    data = generate_mock_data(symbol, periods=100)
    
    # 6. Train the anomaly detector
    console.print(Panel("Training anomaly detector", style="cyan"))
    detector.train(data.iloc[:-20])  # Train on all but the last 20 points
    
    # 7. Simulate trading over the last 20 periods
    console.print(Panel("Beginning trading simulation", style="cyan"))
    
    for i in range(80, 100):
        # Current data window
        current_data = data.iloc[i-10:i+1]
        current_price = current_data['close'].iloc[-1]
        
        console.print(f"\n[bold]Simulation Period {i+1}: Price ${current_price:.2f}[/bold]")
        
        # Detect anomalies
        anomaly_result = detector.detect_anomalies(current_data)
        anomaly_score = anomaly_result.get("latest_score", 0)
        
        console.print(f"Current anomaly score: {anomaly_score:.4f}")
        
        # Update risk parameters based on anomalies
        risk_manager.update_risk_from_anomalies(anomaly_result)
        
        # Display current risk status
        risk_status = risk_manager.get_anomaly_risk_status()
        display_risk_status(risk_status)
        
        # Generate a mock trade decision (simplified for demo)
        direction = "long" if np.random.random() > 0.3 else "short"
        quantity = 100
        entry_price = current_price
        stop_price = entry_price * 0.97 if direction == "long" else entry_price * 1.03
        target_price = entry_price * 1.05 if direction == "long" else entry_price * 0.95
        
        # Try to execute the trade
        console.print(Panel(f"Attempting to execute {direction.upper()} trade", style="magenta"))
        trade = executor.submit_trade(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price
        )
        
        # Wait a moment for readability
        time.sleep(1)

def display_risk_status(risk_status):
    """Display the current anomaly risk status."""
    table = Table(title="Current Anomaly Risk Status")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="yellow")
    
    for key, value in risk_status.items():
        if key == "anomaly_score":
            table.add_row("Anomaly Score", f"{value:.4f}")
        elif key == "risk_level":
            style = "green"
            if value == "moderate":
                style = "yellow"
            elif value == "high":
                style = "orange3"
            elif value == "critical":
                style = "red"
            table.add_row("Risk Level", f"[{style}]{value}[/{style}]")
        elif key == "in_cooldown":
            table.add_row("In Cooldown", "Yes" if value else "No")
        elif key == "cooldown_end" and value:
            remaining = value - datetime.now()
            minutes = remaining.total_seconds() / 60
            table.add_row("Cooldown Remaining", f"{minutes:.1f} minutes")
        elif key == "position_size_modifier":
            table.add_row("Position Size Modifier", f"{value:.2f} ({(1-value)*100:.0f}% reduction)")
        elif key == "stop_loss_modifier":
            table.add_row("Stop Loss Modifier", f"{value:.2f}")
        elif key == "trading_restricted":
            status = "[red]Restricted[/red]" if value else "[green]Allowed[/green]"
            table.add_row("Trading Status", status)
        elif key == "anomaly_types" and value:
            table.add_row("Anomaly Types", ", ".join(value))
    
    console.print(table)

def main():
    """Main function."""
    try:
        simulate_trading_with_anomalies()
    except KeyboardInterrupt:
        console.print("[yellow]Simulation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error in simulation: {str(e)}[/bold red]")
        import traceback
        traceback.print_exc()
    
    console.print("\n[bold green]Simulation completed[/bold green]")
    
if __name__ == "__main__":
    main() 