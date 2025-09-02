"""
Multi-Asset Adapter module for extending trading system to support multiple asset classes.

This module provides adapters for futures, crypto, and forex markets, normalizing data formats
and execution interfaces to work with the existing trading infrastructure.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from enum import Enum
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import required modules
try:
    from trading_bot.data.market_data_provider import MarketDataProvider
    from trading_bot.execution.order_executor import OrderExecutor
    from trading_bot.journal.trade_journal import TradeJournal
    from trading_bot.risk.position_sizer import PositionSizer
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

class AssetClass(Enum):
    """Enum for supported asset classes."""
    EQUITY = "equity"
    FUTURES = "futures" 
    CRYPTO = "crypto"
    FOREX = "forex"
    OPTIONS = "options"

class MultiAssetAdapter:
    """
    Main adapter class that provides consistent interfaces for different asset classes.
    
    This class harmonizes data formats, order execution, position sizing, and risk
    management across different asset types to work with the existing trading system.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        asset_class: AssetClass = AssetClass.EQUITY,
        data_provider: Optional[MarketDataProvider] = None,
        order_executor: Optional[OrderExecutor] = None,
        journal: Optional[TradeJournal] = None,
        position_sizer: Optional[PositionSizer] = None,
    ):
        """
        Initialize the multi-asset adapter.
        
        Args:
            config_path: Path to configuration file
            asset_class: Type of asset to trade
            data_provider: Optional data provider instance
            order_executor: Optional order executor instance
            journal: Optional trade journal instance
            position_sizer: Optional position sizer instance
        """
        self.asset_class = asset_class
        self.config = self._load_config(config_path)
        
        # Initialize components or use provided ones
        self.data_provider = data_provider or self._initialize_data_provider()
        self.order_executor = order_executor or self._initialize_order_executor()
        self.journal = journal or self._initialize_journal()
        self.position_sizer = position_sizer or self._initialize_position_sizer()
        
        # Asset-specific settings
        self.multipliers = self.config.get("multipliers", {})
        self.tick_sizes = self.config.get("tick_sizes", {})
        self.trading_hours = self.config.get("trading_hours", {})
        self.margin_requirements = self.config.get("margin_requirements", {})
        
        logger.info(f"Initialized MultiAssetAdapter for {asset_class.value}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if not config_path or not os.path.exists(config_path):
            logger.warning(f"Config path not provided or doesn't exist: {config_path}")
            return self._get_default_config()
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "multipliers": {
                "ES": 50,  # E-mini S&P 500
                "NQ": 20,  # E-mini Nasdaq-100
                "CL": 1000,  # Crude Oil
                "GC": 100,  # Gold
            },
            "tick_sizes": {
                "ES": 0.25,
                "NQ": 0.25,
                "BTC-USD": 0.01,
                "ETH-USD": 0.01,
                "EUR/USD": 0.0001,
            },
            "trading_hours": {
                "equity": {"start": "09:30", "end": "16:00", "timezone": "America/New_York"},
                "futures": {"start": "18:00", "end": "17:00", "timezone": "America/New_York"},
                "crypto": {"start": "00:00", "end": "23:59", "timezone": "UTC"},
                "forex": {"start": "00:00", "end": "23:59", "timezone": "UTC"},
            },
            "margin_requirements": {
                "futures": {
                    "ES": 12000,
                    "NQ": 15000,
                },
                "crypto": 0.5,  # 50% margin
                "forex": 0.02,  # 50:1 leverage
            }
        }
    
    def _initialize_data_provider(self) -> MarketDataProvider:
        """Initialize the appropriate data provider for the asset class."""
        try:
            provider_config = {
                "asset_class": self.asset_class.value,
                "api_keys": self.config.get("api_keys", {}),
                "endpoints": self.config.get("endpoints", {})
            }
            return MarketDataProvider(**provider_config)
        except Exception as e:
            logger.error(f"Failed to initialize data provider: {e}")
            raise
    
    def _initialize_order_executor(self) -> OrderExecutor:
        """Initialize the appropriate order executor for the asset class."""
        try:
            # First, try to use the broker registry if available
            try:
                from trading_bot.brokers.broker_registry import get_broker_registry
                from trading_bot.brokers.brokerage_client import BrokerageClient
                
                registry = get_broker_registry()
                
                # Look for a broker that matches the asset class
                broker_map = {
                    AssetClass.EQUITY: ['alpaca', 'alpaca_paper', 'ib', 'tradier'],
                    AssetClass.FUTURES: ['ib', 'futures', 'ninja_trader'],
                    AssetClass.CRYPTO: ['alpaca_crypto', 'binance', 'coinbase', 'ftx'],
                    AssetClass.FOREX: ['oanda', 'forex_broker', 'fxcm'],
                    AssetClass.OPTIONS: ['alpaca_options', 'tradier', 'ib_options']
                }
                
                # Get potential broker names for this asset class
                broker_names = broker_map.get(self.asset_class, [])
                
                # Try each broker name
                for name in broker_names:
                    broker = registry.get_broker(name)
                    if broker:
                        logger.info(f"Using {name} broker for {self.asset_class.value}")
                        
                        # Create an adapter between BrokerageClient and OrderExecutor
                        # This allows using the new broker interface with existing code
                        executor_config = {
                            "asset_class": self.asset_class.value,
                            "broker_client": broker
                        }
                        
                        # Create the appropriate executor based on broker type
                        if name.startswith('alpaca'):
                            from trading_bot.execution.order_executor import AlpacaOrderExecutor
                            return AlpacaOrderExecutor(**executor_config)
                        elif name.startswith('tradier'):
                            from trading_bot.execution.order_executor import TradierOrderExecutor
                            return TradierOrderExecutor(**executor_config)
                        else:
                            # For other broker types, use a generic adapter
                            from trading_bot.execution.order_executor import BrokerageClientAdapter
                            return BrokerageClientAdapter(**executor_config)
                
                # If we get here, no suitable broker was found in the registry
                logger.warning(f"No suitable broker found in registry for {self.asset_class.value}")
                
                # Try to create a default broker
                if self.asset_class == AssetClass.EQUITY:
                    # Create and register an Alpaca broker
                    try:
                        from trading_bot.brokers.alpaca_client import AlpacaClient
                        
                        # Get API credentials from config
                        api_key = self.config.get('api_keys', {}).get('alpaca', {}).get('api_key')
                        api_secret = self.config.get('api_keys', {}).get('alpaca', {}).get('api_secret')
                        paper_trading = self.config.get('paper_trading', True)
                        
                        # Create broker client
                        broker_name = 'alpaca_paper' if paper_trading else 'alpaca_live'
                        broker = AlpacaClient(
                            api_key=api_key,
                            api_secret=api_secret,
                            paper_trading=paper_trading
                        )
                        
                        # Register the broker
                        registry.register_broker(broker_name, broker)
                        
                        # Create an executor using this broker
                        from trading_bot.execution.order_executor import AlpacaOrderExecutor
                        return AlpacaOrderExecutor(
                            api_key=api_key,
                            api_secret=api_secret,
                            paper_trading=paper_trading
                        )
                    except Exception as e:
                        logger.error(f"Failed to create Alpaca broker: {e}")
            
            except ImportError:
                # Broker registry not available, fall back to traditional method
                logger.info("Broker registry not available, using traditional OrderExecutor initialization")
                pass
            
            # Fall back to traditional method if broker registry approach didn't work
            executor_config = {
                "asset_class": self.asset_class.value,
                "api_keys": self.config.get("api_keys", {}),
                "endpoints": self.config.get("endpoints", {})
            }
            return OrderExecutor(**executor_config)
        
        except Exception as e:
            logger.error(f"Failed to initialize order executor: {e}")
            raise
    
    def _initialize_journal(self) -> TradeJournal:
        """Initialize the trade journal."""
        try:
            journal_config = {
                "db_path": self.config.get("journal_path", "trading_journal.db"),
                "asset_class": self.asset_class.value
            }
            return TradeJournal(**journal_config)
        except Exception as e:
            logger.error(f"Failed to initialize trade journal: {e}")
            raise
    
    def _initialize_position_sizer(self) -> PositionSizer:
        """Initialize the position sizer."""
        try:
            sizer_config = {
                "asset_class": self.asset_class.value,
                "risk_percentage": self.config.get("risk_percentage", 1.0),
                "max_position_size": self.config.get("max_position_size", {})
            }
            return PositionSizer(**sizer_config)
        except Exception as e:
            logger.error(f"Failed to initialize position sizer: {e}")
            raise
    
    def get_data(
        self, 
        symbol: str, 
        timeframe: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get market data for the specified symbol and timeframe.
        
        Args:
            symbol: Asset symbol to fetch data for
            timeframe: Data timeframe (e.g., "1m", "5m", "1h", "1d")
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            limit: Maximum number of bars to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Normalize symbol based on asset class
            normalized_symbol = self._normalize_symbol(symbol)
            
            # Get data from provider
            data = self.data_provider.get_historical_data(
                symbol=normalized_symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            
            # Process data for the specific asset class
            data = self._process_asset_data(data, symbol)
            
            logger.info(f"Retrieved {len(data)} bars for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Failed to get data for {symbol}: {e}")
            raise
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format based on asset class.
        
        Args:
            symbol: Original symbol
            
        Returns:
            Normalized symbol for the specific data provider
        """
        if self.asset_class == AssetClass.FUTURES:
            # Add exchange code if missing (e.g., ES -> ES=F)
            if not symbol.endswith("=F") and not "/" in symbol:
                return f"{symbol}=F"
        elif self.asset_class == AssetClass.CRYPTO:
            # Add pair suffix if missing (e.g., BTC -> BTC-USD)
            if not "-" in symbol and not "/" in symbol:
                return f"{symbol}-USD"
        elif self.asset_class == AssetClass.FOREX:
            # Ensure forex pair format (e.g., EUR/USD)
            if not "/" in symbol:
                if len(symbol) == 6:
                    return f"{symbol[:3]}/{symbol[3:]}"
        
        return symbol
    
    def _process_asset_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Process data for the specific asset class.
        
        Args:
            data: Original data DataFrame
            symbol: Asset symbol
            
        Returns:
            Processed DataFrame
        """
        if data is None or data.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        # Add contract multiplier for futures
        if self.asset_class == AssetClass.FUTURES:
            base_symbol = symbol.split("=")[0] if "=" in symbol else symbol
            multiplier = self.multipliers.get(base_symbol, 1)
            result["contract_value"] = result["close"] * multiplier
            result["point_value"] = multiplier
            
        # Add tick size
        result["tick_size"] = self.tick_sizes.get(symbol, 0.01)
        
        # Add trading hours
        hours = self.trading_hours.get(self.asset_class.value, {})
        result["trading_start"] = hours.get("start", "00:00")
        result["trading_end"] = hours.get("end", "23:59")
        result["timezone"] = hours.get("timezone", "UTC")
        
        return result
    
    def calculate_position_size(
        self, 
        symbol: str, 
        entry_price: float, 
        stop_price: float,
        account_balance: float,
        risk_percentage: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            symbol: Asset symbol
            entry_price: Planned entry price
            stop_price: Planned stop loss price
            account_balance: Current account balance
            risk_percentage: Risk percentage override (optional)
            
        Returns:
            Dictionary with position sizing information
        """
        try:
            # Use asset-specific logic for position sizing
            if self.asset_class == AssetClass.FUTURES:
                return self._calculate_futures_position(
                    symbol, entry_price, stop_price, account_balance, risk_percentage
                )
            elif self.asset_class == AssetClass.CRYPTO:
                return self._calculate_crypto_position(
                    symbol, entry_price, stop_price, account_balance, risk_percentage
                )
            elif self.asset_class == AssetClass.FOREX:
                return self._calculate_forex_position(
                    symbol, entry_price, stop_price, account_balance, risk_percentage
                )
            else:
                # Default to standard equity position sizing
                return self.position_sizer.calculate_position_size(
                    symbol, entry_price, stop_price, account_balance, risk_percentage
                )
        except Exception as e:
            logger.error(f"Failed to calculate position size for {symbol}: {e}")
            return {
                "position_size": 0,
                "risk_amount": 0,
                "risk_percentage": 0
            }

    def _calculate_futures_position(
        self, 
        symbol: str, 
        entry_price: float, 
        stop_price: float,
        account_balance: float,
        risk_percentage: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculate position size for futures."""
        # Use risk percentage from config if not provided
        risk_pct = risk_percentage or self.config.get("risk_percentage", 1.0)
        
        # Calculate risk amount
        risk_amount = account_balance * (risk_pct / 100)
        
        # Get contract details
        base_symbol = symbol.split("=")[0] if "=" in symbol else symbol
        multiplier = self.multipliers.get(base_symbol, 1)
        
        # Calculate risk per contract
        risk_per_contract = abs(entry_price - stop_price) * multiplier
        
        # Calculate position size in contracts
        contracts = int(risk_amount / risk_per_contract) if risk_per_contract > 0 else 0
        
        # Check margin requirements
        margin_per_contract = self.margin_requirements.get("futures", {}).get(base_symbol, 0)
        max_contracts_by_margin = int(account_balance * 0.5 / margin_per_contract) if margin_per_contract > 0 else 999
        
        # Take the smaller value
        contracts = min(contracts, max_contracts_by_margin)
        
        return {
            "position_size": contracts,
            "contracts": contracts,
            "risk_amount": risk_amount,
            "risk_percentage": risk_pct,
            "multiplier": multiplier,
            "margin_requirement": margin_per_contract,
            "margin_total": contracts * margin_per_contract
        }
    
    def _calculate_crypto_position(
        self, 
        symbol: str, 
        entry_price: float, 
        stop_price: float,
        account_balance: float,
        risk_percentage: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculate position size for crypto."""
        # Use risk percentage from config if not provided
        risk_pct = risk_percentage or self.config.get("risk_percentage", 1.0)
        
        # Calculate risk amount
        risk_amount = account_balance * (risk_pct / 100)
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_price)
        
        # Calculate position size in units
        units = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
        
        # Check margin requirements for leveraged trades
        margin_requirement = self.margin_requirements.get("crypto", 1.0)  # Default to no leverage
        max_units_by_margin = (account_balance * 0.8) / (entry_price * margin_requirement)
        
        # Take the smaller value
        units = min(units, max_units_by_margin)
        
        return {
            "position_size": units,
            "units": units,
            "risk_amount": risk_amount,
            "risk_percentage": risk_pct,
            "notional_value": units * entry_price,
            "margin_requirement": entry_price * units * margin_requirement
        }
    
    def _calculate_forex_position(
        self, 
        symbol: str, 
        entry_price: float, 
        stop_price: float,
        account_balance: float,
        risk_percentage: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculate position size for forex."""
        # Use risk percentage from config if not provided
        risk_pct = risk_percentage or self.config.get("risk_percentage", 1.0)
        
        # Calculate risk amount
        risk_amount = account_balance * (risk_pct / 100)
        
        # Calculate pip value and risk
        pip_size = 0.0001 if not "/JPY" in symbol else 0.01
        pip_risk = abs(entry_price - stop_price) / pip_size
        
        # Calculate standard lot size (100,000 units of base currency)
        lot_size = 100000
        
        # For USD/XXX pairs, pip value = pip_size * lot_size
        # For XXX/USD pairs, pip value = pip_size * lot_size / entry_price
        # For XXX/YYY pairs, need to convert to USD
        
        # Simplified calculation assuming USD account
        if symbol.endswith("/USD"):
            pip_value = pip_size * lot_size / entry_price
        else:
            pip_value = pip_size * lot_size
        
        # Calculate position size in lots
        lots_raw = risk_amount / (pip_risk * pip_value)
        
        # Round to standard lot sizes (0.01 lot increments)
        lots = round(lots_raw * 100) / 100
        
        # Leverage check
        margin_requirement = self.margin_requirements.get("forex", 0.02)  # Default 50:1 leverage
        max_lots_by_margin = (account_balance * 0.8) / (lot_size * entry_price * margin_requirement)
        
        # Take the smaller value
        lots = min(lots, max_lots_by_margin)
        
        return {
            "position_size": lots,
            "lots": lots,
            "units": lots * lot_size,
            "risk_amount": risk_amount,
            "risk_percentage": risk_pct,
            "pip_value": pip_value,
            "pip_risk": pip_risk,
            "notional_value": lots * lot_size * entry_price,
            "margin_requirement": lots * lot_size * entry_price * margin_requirement
        }
    
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: Union[float, int],
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Place an order for the specified asset.
        
        Args:
            symbol: Asset symbol
            side: Order side ("buy" or "sell")
            quantity: Order quantity (contracts, units, or lots)
            order_type: Order type (market, limit, stop, etc.)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force (day, gtc, ioc, etc.)
            **kwargs: Additional asset-specific parameters
            
        Returns:
            Order information
        """
        try:
            # Normalize symbol
            normalized_symbol = self._normalize_symbol(symbol)
            
            # Adjust quantity based on asset class
            adjusted_quantity = self._adjust_quantity(symbol, quantity)
            
            # Place order through executor
            order_result = self.order_executor.place_order(
                symbol=normalized_symbol,
                side=side,
                quantity=adjusted_quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                **kwargs
            )
            
            # Record to journal
            self._record_trade(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=limit_price if order_type == "limit" else None,
                order_type=order_type,
                order_id=order_result.get("order_id")
            )
            
            logger.info(f"Placed {side} order for {quantity} {symbol}")
            return order_result
        except Exception as e:
            logger.error(f"Failed to place order for {symbol}: {e}")
            raise

    def _adjust_quantity(self, symbol: str, quantity: Union[float, int]) -> Union[float, int]:
        """
        Adjust quantity based on asset class.
        
        Args:
            symbol: Asset symbol
            quantity: Original quantity
            
        Returns:
            Adjusted quantity
        """
        if self.asset_class == AssetClass.FUTURES:
            # Futures are always integer contracts
            return int(quantity)
        elif self.asset_class == AssetClass.FOREX:
            # Convert lots to units
            if quantity < 100:  # Assuming quantity is in lots if < 100
                return quantity * 100000  # Standard lot = 100,000 units
        
        return quantity
    
    def _record_trade(
        self,
        symbol: str,
        side: str,
        quantity: Union[float, int],
        entry_price: Optional[float] = None,
        order_type: str = "market",
        order_id: Optional[str] = None
    ) -> None:
        """Record trade in the journal."""
        try:
            trade_data = {
                "symbol": symbol,
                "asset_class": self.asset_class.value,
                "side": side,
                "quantity": quantity,
                "entry_price": entry_price,
                "order_type": order_type,
                "order_id": order_id,
                "timestamp": datetime.now().isoformat()
            }
            
            self.journal.record_trade(trade_data)
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            return self.order_executor.get_account_info()
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}
            
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        try:
            return self.order_executor.get_positions()
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        try:
            return self.order_executor.get_orders(status=status)
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

if __name__ == "__main__":
    # Example usage
    adapter = MultiAssetAdapter(
        asset_class=AssetClass.FUTURES,
        config_path="config/multi_asset_config.json"
    )
    
    # Get data example
    data = adapter.get_data(
        symbol="ES",
        timeframe="1h",
        limit=10
    )
    print(f"Data shape: {data.shape}")
    
    # Calculate position size example
    position = adapter.calculate_position_size(
        symbol="ES", 
        entry_price=4500, 
        stop_price=4480,
        account_balance=100000,
        risk_percentage=1.0
    )
    print(f"Position size: {position}") 