import pytest

from trading_bot import (
    TradingBot,
    TradingMode,
    TradeResult,
    TradeAction,
    TradeType,
    MainOrchestrator,
    BenBotAssistant
)

def test_core_imports_exist():
    """Test that core trading bot components can be imported"""
    assert TradingMode is not None
    assert TradeResult is not None
    assert TradeAction is not None
    assert TradeType is not None
    assert TradingBot is not None
    assert MainOrchestrator is not None
    assert BenBotAssistant is not None

def test_trading_mode_values():
    """Test that TradingMode enum has expected values"""
    assert TradingMode.LIVE == "live"
    assert TradingMode.PAPER == "paper"
    assert TradingMode.BACKTEST == "backtest"
    assert TradingMode.SIMULATION == "simulation"

def test_trade_result_values():
    """Test that TradeResult enum has expected values"""
    assert TradeResult.SUCCESS == "success"
    assert TradeResult.PARTIAL == "partial"
    assert TradeResult.FAILED == "failed"
    assert TradeResult.PENDING == "pending"

def test_trade_action_values():
    """Test that TradeAction enum has expected values"""
    assert TradeAction.BUY == "buy"
    assert TradeAction.SELL == "sell"
    assert TradeAction.SHORT == "short"
    assert TradeAction.COVER == "cover"

def test_trade_type_values():
    """Test that TradeType enum has expected values"""
    assert TradeType.ENTRY == "entry"
    assert TradeType.EXIT == "exit"
    assert TradeType.ADJUSTMENT == "adjustment"
    assert TradeType.STOP_LOSS == "stop_loss"
    assert TradeType.TAKE_PROFIT == "take_profit" 