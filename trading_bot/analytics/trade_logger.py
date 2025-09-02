import os
import csv
import json
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class TradeLogger:
    """
    Records and analyzes trade data for performance tracking.
    """
    
    def __init__(self, log_dir="logs"):
        """Initialize the trade logger with a directory for log files."""
        self.log_dir = log_dir
        self.trades_file = os.path.join(log_dir, "trades.csv")
        self.positions_file = os.path.join(log_dir, "positions.csv")
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Create trades file with header if it doesn't exist
        if not os.path.exists(self.trades_file):
            with open(self.trades_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "order_id", "strategy", "symbol", "asset_type", 
                    "action", "quantity", "price", "commission", "status"
                ])
        
        # Create positions file with header if it doesn't exist
        if not os.path.exists(self.positions_file):
            with open(self.positions_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "symbol", "asset_type", "quantity", "entry_price", "entry_date",
                    "exit_price", "exit_date", "pnl", "pnl_percent", "strategy", "status"
                ])
    
    def log_trade(self, trade_data):
        """
        Log a trade to the trades CSV file.
        
        Args:
            trade_data (dict): Trade data including:
                - timestamp (str, optional): Trade timestamp, defaults to current time
                - order_id (str): Broker order ID
                - strategy (str): Strategy that generated the trade
                - symbol (str): Stock or option symbol
                - asset_type (str): 'equity' or 'option'
                - action (str): 'buy', 'sell', 'buy_to_open', etc.
                - quantity (int): Number of shares/contracts
                - price (float): Execution price
                - commission (float, optional): Commission paid
                - status (str, optional): Order status, defaults to 'filled'
        """
        # Set defaults
        timestamp = trade_data.get("timestamp", datetime.now().isoformat())
        commission = trade_data.get("commission", 0.0)
        status = trade_data.get("status", "filled")
        
        # Write trade to CSV
        with open(self.trades_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                trade_data.get("order_id", ""),
                trade_data.get("strategy", ""),
                trade_data.get("symbol", ""),
                trade_data.get("asset_type", ""),
                trade_data.get("action", ""),
                trade_data.get("quantity", 0),
                trade_data.get("price", 0.0),
                commission,
                status
            ])
        
        logger.info(f"Logged trade: {json.dumps(trade_data)}")
        
        # Update positions based on the trade
        self._update_positions(trade_data)
    
    def _update_positions(self, trade_data):
        """
        Update the positions file based on a new trade.
        
        This handles opening new positions, adding to existing positions,
        or closing positions (partially or fully).
        """
        symbol = trade_data.get("symbol", "")
        asset_type = trade_data.get("asset_type", "")
        action = trade_data.get("action", "")
        quantity = trade_data.get("quantity", 0)
        price = trade_data.get("price", 0.0)
        strategy = trade_data.get("strategy", "")
        timestamp = trade_data.get("timestamp", datetime.now().isoformat())
        
        # Skip if essential data is missing
        if not all([symbol, asset_type, action, quantity, price]):
            logger.warning("Missing essential trade data, skipping position update")
            return
        
        # Load current positions
        positions = []
        if os.path.exists(self.positions_file):
            try:
                positions_df = pd.read_csv(self.positions_file)
                positions = positions_df.to_dict(orient="records")
            except Exception as e:
                logger.error(f"Error reading positions file: {str(e)}")
                return
        
        # Find existing position for this symbol
        existing_position = None
        for position in positions:
            if (position["symbol"] == symbol and 
                position["asset_type"] == asset_type and 
                position["status"] == "open"):
                existing_position = position
                break
        
        # Process based on action
        if action in ["buy", "buy_to_open"]:
            if existing_position:
                # Add to existing position (average down/up)
                total_quantity = existing_position["quantity"] + quantity
                total_cost = (existing_position["quantity"] * existing_position["entry_price"] + 
                             quantity * price)
                
                # Update average entry price
                existing_position["entry_price"] = total_cost / total_quantity
                existing_position["quantity"] = total_quantity
                existing_position["strategy"] = f"{existing_position['strategy']},{strategy}"
            else:
                # Create new position
                new_position = {
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "quantity": quantity,
                    "entry_price": price,
                    "entry_date": timestamp,
                    "exit_price": 0.0,
                    "exit_date": "",
                    "pnl": 0.0,
                    "pnl_percent": 0.0,
                    "strategy": strategy,
                    "status": "open"
                }
                positions.append(new_position)
        
        elif action in ["sell", "sell_to_close"]:
            if existing_position:
                # Close position (fully or partially)
                if quantity >= existing_position["quantity"]:
                    # Fully close
                    pnl = (price - existing_position["entry_price"]) * existing_position["quantity"]
                    pnl_percent = (price / existing_position["entry_price"] - 1) * 100
                    
                    existing_position["exit_price"] = price
                    existing_position["exit_date"] = timestamp
                    existing_position["pnl"] = pnl
                    existing_position["pnl_percent"] = pnl_percent
                    existing_position["status"] = "closed"
                else:
                    # Partially close - create a closed position and update the open one
                    closed_quantity = quantity
                    remaining_quantity = existing_position["quantity"] - closed_quantity
                    
                    # Calculate P&L for the closed portion
                    pnl = (price - existing_position["entry_price"]) * closed_quantity
                    pnl_percent = (price / existing_position["entry_price"] - 1) * 100
                    
                    # Create a new entry for the closed portion
                    closed_position = {
                        "symbol": symbol,
                        "asset_type": asset_type,
                        "quantity": closed_quantity,
                        "entry_price": existing_position["entry_price"],
                        "entry_date": existing_position["entry_date"],
                        "exit_price": price,
                        "exit_date": timestamp,
                        "pnl": pnl,
                        "pnl_percent": pnl_percent,
                        "strategy": existing_position["strategy"],
                        "status": "closed"
                    }
                    positions.append(closed_position)
                    
                    # Update the remaining open position
                    existing_position["quantity"] = remaining_quantity
            else:
                logger.warning(f"Attempted to close non-existent position: {symbol}")
        
        # Save updated positions
        pd.DataFrame(positions).to_csv(self.positions_file, index=False)
    
    def get_open_positions(self):
        """
        Get all currently open positions.
        
        Returns:
            list: List of dictionaries representing open positions
        """
        if not os.path.exists(self.positions_file):
            return []
        
        try:
            positions_df = pd.read_csv(self.positions_file)
            open_positions = positions_df[positions_df["status"] == "open"].to_dict(orient="records")
            return open_positions
        except Exception as e:
            logger.error(f"Error reading positions file: {str(e)}")
            return []
    
    def get_trade_history(self, symbol=None, strategy=None, days=None):
        """
        Get trade history with optional filtering.
        
        Args:
            symbol (str, optional): Filter by symbol
            strategy (str, optional): Filter by strategy
            days (int, optional): Filter by number of recent days
        
        Returns:
            pandas.DataFrame: Filtered trade history
        """
        if not os.path.exists(self.trades_file):
            return pd.DataFrame()
        
        try:
            trades_df = pd.read_csv(self.trades_file)
            
            # Convert timestamp to datetime
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
            
            # Apply filters
            if symbol:
                trades_df = trades_df[trades_df["symbol"] == symbol]
            
            if strategy:
                trades_df = trades_df[trades_df["strategy"] == strategy]
            
            if days:
                cutoff_date = datetime.now() - pd.Timedelta(days=days)
                trades_df = trades_df[trades_df["timestamp"] >= cutoff_date]
            
            return trades_df
        except Exception as e:
            logger.error(f"Error reading trades file: {str(e)}")
            return pd.DataFrame()
    
    def calculate_performance_metrics(self):
        """
        Calculate performance metrics for all closed trades.
        
        Returns:
            dict: Dictionary of performance metrics
        """
        if not os.path.exists(self.positions_file):
            return {}
        
        try:
            positions_df = pd.read_csv(self.positions_file)
            closed_positions = positions_df[positions_df["status"] == "closed"]
            
            if closed_positions.empty:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "average_pnl": 0.0,
                    "average_win": 0.0,
                    "average_loss": 0.0,
                    "largest_win": 0.0,
                    "largest_loss": 0.0,
                    "profit_factor": 0.0
                }
            
            # Basic metrics
            total_trades = len(closed_positions)
            winning_trades = len(closed_positions[closed_positions["pnl"] > 0])
            losing_trades = len(closed_positions[closed_positions["pnl"] < 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            total_pnl = closed_positions["pnl"].sum()
            average_pnl = closed_positions["pnl"].mean()
            
            # Advanced metrics
            average_win = closed_positions[closed_positions["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0
            average_loss = closed_positions[closed_positions["pnl"] < 0]["pnl"].mean() if losing_trades > 0 else 0
            
            largest_win = closed_positions["pnl"].max()
            largest_loss = closed_positions["pnl"].min()
            
            gross_profit = closed_positions[closed_positions["pnl"] > 0]["pnl"].sum()
            gross_loss = abs(closed_positions[closed_positions["pnl"] < 0]["pnl"].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "average_pnl": average_pnl,
                "average_win": average_win,
                "average_loss": average_loss,
                "largest_win": largest_win,
                "largest_loss": largest_loss,
                "profit_factor": profit_factor
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {} 