"""
Test script for E*TRADE Extensions

This script tests the E*TRADE specific extensions (AdvancedOptionsExtension and
PortfolioAnalysisExtension) with a mock E*TRADE client.
"""

import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from trading_bot.brokers.extensions.etrade_extensions import (
    ETradeOptionsExtension,
    ETradePortfolioExtension
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create a mock E*TRADE client for testing
class MockETradeClient:
    """
    Mock E*TRADE client that returns test data for extension testing
    """
    
    def get_option_chains(self, symbol, expiration=None):
        """Mock option chain data"""
        logger.info(f"Mock: Getting option chains for {symbol} with expiration {expiration}")
        return {
            "calls": [
                {"strikePrice": 95.0, "symbol": f"{symbol}240621C00095000", "bid": 7.5, "ask": 7.8},
                {"strikePrice": 100.0, "symbol": f"{symbol}240621C00100000", "bid": 4.2, "ask": 4.5},
                {"strikePrice": 105.0, "symbol": f"{symbol}240621C00105000", "bid": 2.1, "ask": 2.3},
                {"strikePrice": 110.0, "symbol": f"{symbol}240621C00110000", "bid": 0.8, "ask": 1.0}
            ],
            "puts": [
                {"strikePrice": 95.0, "symbol": f"{symbol}240621P00095000", "bid": 0.7, "ask": 0.9},
                {"strikePrice": 100.0, "symbol": f"{symbol}240621P00100000", "bid": 2.0, "ask": 2.2},
                {"strikePrice": 105.0, "symbol": f"{symbol}240621P00105000", "bid": 4.0, "ask": 4.3},
                {"strikePrice": 110.0, "symbol": f"{symbol}240621P00110000", "bid": 7.2, "ask": 7.5}
            ]
        }
    
    def get_option_expiration_dates(self, symbol):
        """Mock option expiration dates"""
        logger.info(f"Mock: Getting option expiration dates for {symbol}")
        today = datetime.now()
        return [
            (today + timedelta(days=30)).strftime("%Y%m%d"),
            (today + timedelta(days=60)).strftime("%Y%m%d"),
            (today + timedelta(days=90)).strftime("%Y%m%d")
        ]
    
    def get_quote(self, symbol):
        """Mock quote data"""
        logger.info(f"Mock: Getting quote for {symbol}")
        return {"last": 102.50, "bid": 102.45, "ask": 102.55, "volume": 1500000}
    
    def place_option_spread_order(self, params):
        """Mock order placement"""
        logger.info(f"Mock: Placing option spread order: {params}")
        return {
            "order_id": "ORD12345",
            "status": "OPEN",
            "created_at": datetime.now().isoformat(),
            "params": params
        }
    
    def get_positions(self):
        """Mock portfolio positions"""
        logger.info("Mock: Getting portfolio positions")
        return [
            {"symbol": "AAPL", "quantity": 100, "costBasis": 15000, "marketValue": 17500, "purchaseDate": datetime.now() - timedelta(days=120)},
            {"symbol": "MSFT", "quantity": 50, "costBasis": 12000, "marketValue": 13000, "purchaseDate": datetime.now() - timedelta(days=90)},
            {"symbol": "AMZN", "quantity": 25, "costBasis": 25000, "marketValue": 28000, "purchaseDate": datetime.now() - timedelta(days=60)}
        ]
    
    def get_historical_prices(self, symbol, days=None, start=None, end=None, interval=None):
        """Mock historical price data"""
        logger.info(f"Mock: Getting historical prices for {symbol} (days={days}, interval={interval})")
        
        # Generate realistic price series with some randomness
        np.random.seed(hash(symbol) % 100)  # Use symbol to seed for consistent results per symbol
        
        if days:
            length = days
        elif start and end:
            length = (end - start).days
        else:
            length = 100
            
        # Generate baseline trend
        trend = np.linspace(0, 0.2, length)  # 20% growth over the period
        
        # Add volatility
        volatility = 0.01  # 1% daily volatility
        random_walk = np.random.normal(0, volatility, length).cumsum()
        
        # Create price series starting at 100
        base_price = 100.0
        prices = base_price * (1 + trend + random_walk)
        
        return prices.tolist()
    
    def get_fundamental_data(self, symbol):
        """Mock fundamental data"""
        logger.info(f"Mock: Getting fundamental data for {symbol}")
        
        # Return different betas based on symbol for more realistic testing
        betas = {
            "AAPL": 1.2,
            "MSFT": 1.1,
            "AMZN": 1.4,
            "DEFAULT": 1.0
        }
        
        return {"beta": betas.get(symbol, betas["DEFAULT"])}


def test_options_extension():
    """Test the E*TRADE options extension functionality"""
    print("\n=== Testing ETradeOptionsExtension ===\n")
    
    client = MockETradeClient()
    extension = ETradeOptionsExtension(client)
    
    # Test option chain retrieval
    symbol = "AAPL"
    chain = extension.get_option_chain(symbol)
    print(f"Option chain for {symbol}: {len(chain['calls'])} calls, {len(chain['puts'])} puts")
    
    # Test expiration dates retrieval
    exp_dates = extension.get_option_expiration_dates(symbol)
    print(f"Expiration dates for {symbol}: {[d.strftime('%Y-%m-%d') for d in exp_dates]}")
    
    # Test option strikes retrieval
    exp_date = datetime.now() + timedelta(days=30)
    strikes = extension.get_option_strikes(symbol, exp_date)
    print(f"Available strikes for {symbol} on {exp_date.strftime('%Y-%m-%d')}: {strikes}")
    
    # Test vertical spread creation
    vertical_spread = extension.create_option_spread(
        symbol,
        "vertical",
        exp_date,
        5.0,  # width
        True,  # bullish
        1      # quantity
    )
    print(f"Vertical spread creation result: Order ID {vertical_spread.get('order_id')}")
    
    # Test iron condor spread creation
    iron_condor = extension.create_option_spread(
        symbol,
        "iron_condor",
        exp_date,
        5.0,  # width
        True,  # direction (doesn't matter much for iron condor)
        1      # quantity
    )
    print(f"Iron condor creation result: Order ID {iron_condor.get('order_id')}")
    
    # Test butterfly spread creation
    butterfly = extension.create_option_spread(
        symbol,
        "butterfly",
        exp_date,
        5.0,  # width
        True,  # call butterfly
        1      # quantity
    )
    print(f"Butterfly spread creation result: Order ID {butterfly.get('order_id')}")
    
    # Test capabilities
    capabilities = extension.get_capabilities()
    print(f"Extension capabilities: {capabilities}")
    
    return extension  # Return for reuse in integration tests


def test_portfolio_extension():
    """Test the E*TRADE portfolio extension functionality"""
    print("\n=== Testing ETradePortfolioExtension ===\n")
    
    client = MockETradeClient()
    extension = ETradePortfolioExtension(client)
    
    # Test portfolio risk metrics
    risk_metrics = extension.get_portfolio_risk_metrics()
    print(f"Portfolio risk metrics: {risk_metrics}")
    
    # Test position performance
    performance = extension.get_position_performance()
    print(f"Position performance summary:")
    print(f"- Symbols: {performance['symbol'].tolist() if not performance.empty else []}")
    print(f"- Total market value: ${performance['market_value'].sum() if not performance.empty else 0:.2f}")
    
    # Test for a specific symbol
    symbol = "AAPL"
    symbol_performance = extension.get_position_performance(symbol=symbol)
    print(f"{symbol} performance: {symbol_performance.to_dict('records') if not symbol_performance.empty else []}")
    
    # Test correlation matrix
    correlation = extension.get_portfolio_correlation_matrix()
    print(f"Correlation matrix shape: {correlation.shape}")
    if not correlation.empty:
        print("Correlation matrix preview:")
        print(correlation.round(2))
    
    # Test capabilities
    capabilities = extension.get_capabilities()
    print(f"Extension capabilities: {capabilities}")
    
    return extension  # Return for reuse in integration tests


def test_event_integration():
    """Test integration with the event system"""
    print("\n=== Testing Event System Integration ===\n")
    
    # This is a simplified test since we can't easily observe the events
    # In a real environment, we would register event listeners
    
    client = MockETradeClient()
    options_ext = ETradeOptionsExtension(client)
    portfolio_ext = ETradePortfolioExtension(client)
    
    # Events that should be published:
    print("The following events should be published to the event bus:")
    print("1. option_chain_retrieved - When getting option chains")
    print("2. option_spread_created - When creating option spreads")
    print("3. portfolio_risk_calculated - When calculating risk metrics")
    print("4. position_performance_calculated - When analyzing position performance")
    print("5. portfolio_correlation_calculated - When generating correlation matrix")
    
    # In real usage, other components would subscribe to these events
    # and react accordingly (e.g., UI updates, alerts, etc.)
    

if __name__ == "__main__":
    print("=== E*TRADE Extensions Test Script ===")
    
    # Test individual extensions
    options_extension = test_options_extension()
    portfolio_extension = test_portfolio_extension()
    
    # Test event integration
    test_event_integration()
    
    print("\n=== All Tests Completed ===\n")
