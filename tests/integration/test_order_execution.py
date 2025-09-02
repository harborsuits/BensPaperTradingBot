import os
import pytest
import pandas as pd
from datetime import datetime, timedelta
import vcr

from trading_bot.config.typed_settings import TradingBotSettings, load_config
from trading_bot.brokers.tradier.client import TradierClient
from trading_bot.brokers.trade_executor import TradeExecutor
from trading_bot.risk.risk_check import RiskManager
from trading_bot.strategies.models import Signal, Trade, Position

# Configure VCR to filter out sensitive information
my_vcr = vcr.VCR(
    cassette_library_dir='tests/integration/cassettes',
    filter_headers=['Authorization'],
    filter_query_parameters=['access_token'],
    record_mode='once',
    match_on=['uri', 'method']
)

# Skip tests if credentials are not available
pytestmark = pytest.mark.skipif(
    os.environ.get('TRADIER_API_KEY') is None or os.environ.get('TRADIER_ACCOUNT_ID') is None,
    reason="Tradier credentials required for integration tests"
)


@pytest.fixture
def settings():
    """Load typed settings for testing."""
    config_path = os.path.join(os.path.dirname(__file__), "../..", "config.yaml")
    if os.path.exists(config_path):
        return load_config(config_path)
    else:
        # Create minimal test settings
        return TradingBotSettings.parse_obj({
            "broker": {
                "name": "tradier",
                "paper_trading": True,
                "account_id": os.environ.get('TRADIER_ACCOUNT_ID', ''),
                "api_key": os.environ.get('TRADIER_API_KEY', '')
            },
            "risk": {
                "max_position_pct": 0.1,
                "portfolio_stop_loss_pct": 0.15,
                "max_drawdown_pct": 0.20,
                "max_positions": 10,
                "max_single_order_size": 5000,
                "enable_max_daily_loss": True,
                "max_daily_loss_pct": 0.05,
                "max_correlation": 0.7
            }
        })


@pytest.fixture
def tradier_client(settings):
    """Create a Tradier client instance."""
    return TradierClient(
        api_key=settings.broker.api_key,
        account_id=settings.broker.account_id,
        paper_trading=True
    )


@pytest.fixture
def risk_manager(settings):
    """Create a RiskManager instance."""
    return RiskManager(settings.risk)


@pytest.fixture
def trade_executor(settings, tradier_client, risk_manager):
    """Create a TradeExecutor instance."""
    return TradeExecutor(
        broker_client=tradier_client,
        risk_manager=risk_manager,
        settings=settings
    )


@pytest.fixture
def sample_signals():
    """Create sample trading signals for testing."""
    return [
        Signal(
            symbol="AAPL",
            direction="buy",
            entry_price=170.50,
            stop_loss=165.00,
            take_profit=180.00,
            confidence=0.75,
            strategy="momentum",
            timestamp=datetime.now(),
            indicators={
                "rsi": 65.5,
                "macd": 2.1,
                "ma_crossover": "bullish"
            }
        ),
        Signal(
            symbol="MSFT",
            direction="buy",
            entry_price=335.20,
            stop_loss=325.00,
            take_profit=350.00,
            confidence=0.8,
            strategy="trend_following",
            timestamp=datetime.now(),
            indicators={
                "rsi": 62.3,
                "macd": 1.8,
                "ma_crossover": "bullish"
            }
        )
    ]


@my_vcr.use_cassette('order_validation.yaml')
def test_order_validation(trade_executor, sample_signals):
    """Test order validation in the TradeExecutor."""
    # Validate orders generated from signals
    for signal in sample_signals:
        order = trade_executor.prepare_order_from_signal(signal)
        assert order is not None
        assert order.symbol == signal.symbol
        assert order.direction == signal.direction
        assert order.limit_price is not None
        assert order.stop_loss == signal.stop_loss
        assert order.take_profit == signal.take_profit
        assert order.validity_in_days > 0

        # Test validation checks
        validated = trade_executor.validate_order(order)
        assert validated is True


@my_vcr.use_cassette('risk_checks.yaml')
def test_risk_checks(trade_executor, risk_manager, sample_signals):
    """Test risk checks for trade execution."""
    # Create initial portfolio context
    portfolio = {
        "buying_power": 100000.0,
        "total_equity": 100000.0,
        "positions": {
            "NVDA": Position(
                symbol="NVDA",
                quantity=50,
                entry_price=750.0,
                current_price=770.0,
                timestamp=datetime.now()
            )
        },
        "trades_today": [
            Trade(
                symbol="NVDA",
                quantity=50,
                price=750.0,
                timestamp=datetime.now() - timedelta(hours=3),
                direction="buy",
                status="filled",
                order_id="12345",
                strategy="momentum"
            )
        ]
    }
    
    # Set portfolio context
    trade_executor.portfolio = portfolio
    
    # Test position sizing
    for signal in sample_signals:
        # Calculate position size based on risk settings
        position_size = trade_executor.calculate_position_size(
            signal=signal,
            portfolio_value=portfolio["total_equity"]
        )
        
        # Verify position size respects max position percentage
        assert position_size * signal.entry_price <= portfolio["total_equity"] * risk_manager.settings.max_position_pct
        
        # Verify position size respects max single order size
        assert position_size * signal.entry_price <= risk_manager.settings.max_single_order_size
        
        # Create order with calculated position size
        order = trade_executor.prepare_order_from_signal(signal, quantity=position_size)
        
        # Perform pre-trade risk checks
        risk_check_result = risk_manager.pre_trade_check(
            order=order,
            portfolio=portfolio
        )
        
        # Verify risk check passes
        assert risk_check_result["approved"] is True
        
        # Verify risk check details
        assert "position_size_pct" in risk_check_result
        assert risk_check_result["position_size_pct"] <= risk_manager.settings.max_position_pct
        assert "max_positions_check" in risk_check_result
        assert risk_check_result["max_positions_check"] is True


@my_vcr.use_cassette('position_correlation.yaml')
def test_position_correlation(risk_manager):
    """Test correlation checks in risk management."""
    # Create mock portfolio with correlated positions
    portfolio = {
        "buying_power": 100000.0,
        "total_equity": 100000.0,
        "positions": {
            "SPY": Position(
                symbol="SPY",
                quantity=100,
                entry_price=450.0,
                current_price=455.0,
                timestamp=datetime.now()
            ),
            "QQQ": Position(
                symbol="QQQ",
                quantity=50,
                entry_price=380.0,
                current_price=385.0,
                timestamp=datetime.now()
            )
        }
    }
    
    # Test new position that would be highly correlated with existing positions
    test_order = {
        "symbol": "IVV",  # S&P 500 ETF (highly correlated with SPY)
        "direction": "buy",
        "quantity": 50,
        "limit_price": 460.0
    }
    
    # Mock correlation data (this would normally come from data provider)
    # Format: {symbol_pair: correlation_value}
    correlation_data = {
        "SPY_IVV": 0.98,  # Very high correlation between SPY and IVV
        "QQQ_IVV": 0.85,  # High correlation between QQQ and IVV
        "SPY_QQQ": 0.85,  # High correlation between SPY and QQQ
    }
    
    # Override the get_correlation method in risk manager for testing
    def mock_get_correlation(symbol1, symbol2):
        pair_key = f"{symbol1}_{symbol2}" if symbol1 < symbol2 else f"{symbol2}_{symbol1}"
        return correlation_data.get(pair_key, 0.0)
    
    risk_manager._get_correlation = mock_get_correlation
    
    # Perform correlation check
    correlation_check = risk_manager._check_correlation_limits(
        symbol=test_order["symbol"],
        portfolio=portfolio
    )
    
    # Verify correlation check fails due to high correlation
    assert correlation_check["passed"] is False
    assert correlation_check["max_correlation"] > risk_manager.settings.max_correlation
    assert "SPY" in correlation_check["correlated_symbols"]


@my_vcr.use_cassette('max_drawdown.yaml')
def test_max_drawdown_protection(risk_manager):
    """Test maximum drawdown protection."""
    # Create portfolio in drawdown
    initial_equity = 100000.0
    current_equity = 82000.0  # 18% drawdown
    
    portfolio = {
        "buying_power": current_equity,
        "total_equity": current_equity,
        "initial_equity": initial_equity,
        "positions": {},
        "drawdown_pct": (initial_equity - current_equity) / initial_equity
    }
    
    # Test new buy order
    test_order = {
        "symbol": "AAPL",
        "direction": "buy",
        "quantity": 10,
        "limit_price": 170.0
    }
    
    # Check drawdown limit
    drawdown_check = risk_manager._check_max_drawdown(
        portfolio=portfolio
    )
    
    # Verify drawdown check fails as we're near max drawdown
    assert drawdown_check["passed"] is False
    assert drawdown_check["current_drawdown"] > risk_manager.settings.max_drawdown_pct
    assert drawdown_check["message"] == "Maximum drawdown limit exceeded"
    
    # Full pre-trade check should also fail
    risk_check_result = risk_manager.pre_trade_check(
        order=test_order,
        portfolio=portfolio
    )
    
    assert risk_check_result["approved"] is False
    assert "max_drawdown" in risk_check_result["failed_checks"]


if __name__ == "__main__":
    pytest.main(["-v", "test_order_execution.py"])
