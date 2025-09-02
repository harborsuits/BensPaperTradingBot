#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Signal Sets

This module provides standardized sets of test signals for different market conditions,
asset classes, and signal sources. These can be used with the end-to-end tester to
validate the trading system's ability to handle various signal patterns.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional


class TestSignalFactory:
    """Factory for creating standardized test signals for different sources."""
    
    @staticmethod
    def tradingview_entry_signal(
        symbol: str,
        direction: str,
        price: float,
        timeframe: str = "1h",
        strategy: str = "Test Strategy",
        confidence: float = 0.8,
        add_indicators: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a TradingView entry signal.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction ("buy" or "sell")
            price: Signal price
            timeframe: Chart timeframe
            strategy: Strategy name
            confidence: Signal confidence
            add_indicators: Whether to add indicator values
            
        Returns:
            Signal data dictionary
        """
        # Base signal data
        signal = {
            "symbol": symbol,
            "action": direction,
            "price": price,
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "strategy": strategy,
            "source": "tradingview",
            "confidence": confidence
        }
        
        # Add stop loss and take profit if provided
        if "stop_loss" in kwargs:
            signal["stop_loss"] = kwargs["stop_loss"]
        
        if "take_profit" in kwargs:
            signal["take_profit"] = kwargs["take_profit"]
        
        # Add indicator values if requested
        if add_indicators:
            if direction == "buy":
                signal["indicators"] = {
                    "rsi": random.uniform(60, 75),
                    "macd": random.uniform(0.1, 0.5),
                    "ema_20": price * 0.995,
                    "ema_50": price * 0.99,
                    "atr": price * 0.005
                }
            else:  # sell
                signal["indicators"] = {
                    "rsi": random.uniform(25, 40),
                    "macd": random.uniform(-0.5, -0.1),
                    "ema_20": price * 1.005,
                    "ema_50": price * 1.01,
                    "atr": price * 0.005
                }
        
        # Add any additional custom fields
        for key, value in kwargs.items():
            if key not in ["stop_loss", "take_profit"]:
                signal[key] = value
        
        return signal
    
    @staticmethod
    def tradingview_exit_signal(
        symbol: str,
        price: float,
        reason: str = "take_profit",
        timeframe: str = "1h",
        strategy: str = "Test Strategy",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a TradingView exit signal.
        
        Args:
            symbol: Trading symbol
            price: Signal price
            reason: Exit reason
            timeframe: Chart timeframe
            strategy: Strategy name
            
        Returns:
            Signal data dictionary
        """
        signal = {
            "symbol": symbol,
            "action": "exit",
            "price": price,
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "strategy": strategy,
            "source": "tradingview",
            "exit_reason": reason
        }
        
        # Add profit information if provided
        if "profit_pips" in kwargs:
            signal["profit_pips"] = kwargs["profit_pips"]
        
        if "profit_percent" in kwargs:
            signal["profit_percent"] = kwargs["profit_percent"]
        
        # Add any additional custom fields
        for key, value in kwargs.items():
            if key not in ["profit_pips", "profit_percent"]:
                signal[key] = value
        
        return signal
    
    @staticmethod
    def alpaca_order_filled(
        symbol: str,
        side: str,
        price: float,
        quantity: int = 10,
        order_type: str = "market",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create an Alpaca order filled signal.
        
        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            price: Fill price
            quantity: Order quantity
            order_type: Order type
            
        Returns:
            Simulated Alpaca order data
        """
        order_id = kwargs.get("order_id", f"test-order-{random.randint(10000, 99999)}")
        client_order_id = kwargs.get("client_order_id", f"test-client-{random.randint(10000, 99999)}")
        
        order = {
            "symbol": symbol,
            "side": side,
            "status": "filled",
            "filled_avg_price": price,
            "qty": quantity,
            "filled_qty": quantity,
            "id": order_id,
            "client_order_id": client_order_id,
            "type": order_type,
            "created_at": (datetime.now() - timedelta(minutes=5)).isoformat(),
            "filled_at": datetime.now().isoformat(),
            "position_effect": kwargs.get("position_effect", "open" if side == "buy" else "close")
        }
        
        # Add any additional custom fields
        for key, value in kwargs.items():
            if key not in ["order_id", "client_order_id", "position_effect"]:
                order[key] = value
        
        return order
    
    @staticmethod
    def finnhub_trade(
        symbol: str,
        price: float,
        volume: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a Finnhub trade signal.
        
        Args:
            symbol: Trading symbol
            price: Trade price
            volume: Trade volume
            
        Returns:
            Simulated Finnhub trade data
        """
        timestamp = kwargs.get("timestamp", int(datetime.now().timestamp() * 1000))
        
        trade = {
            "s": symbol,  # Symbol
            "p": price,   # Price
            "v": volume,  # Volume
            "t": timestamp,  # Timestamp in milliseconds
            "c": kwargs.get("conditions", [])  # Trade conditions
        }
        
        return trade
    
    @staticmethod
    def pattern_detection(
        symbol: str,
        pattern_type: str,
        direction: str,
        price: float,
        confidence: float = 0.8,
        timeframe: str = "1h",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a pattern detection signal.
        
        Args:
            symbol: Trading symbol
            pattern_type: Pattern type
            direction: Pattern direction
            price: Current price
            confidence: Pattern confidence
            timeframe: Chart timeframe
            
        Returns:
            Pattern detection data
        """
        pattern = {
            "symbol": symbol,
            "pattern_type": pattern_type,
            "direction": direction,
            "confidence": confidence,
            "price": price,
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe
        }
        
        # Add pattern-specific data
        if pattern_type == "pin_bar":
            pattern.update({
                "shadow_ratio": random.uniform(2.5, 4.0),
                "body_position": "bottom" if direction == "long" else "top",
                "confirmation_candle": kwargs.get("confirmation_candle", True)
            })
        
        elif pattern_type == "engulfing":
            pattern.update({
                "size_ratio": random.uniform(1.2, 2.0),
                "trend_context": kwargs.get("trend_context", "reversal")
            })
        
        elif pattern_type == "double_top" or pattern_type == "double_bottom":
            pattern.update({
                "height": price * 0.01,
                "width": random.randint(5, 15),
                "symmetry": random.uniform(0.8, 0.95)
            })
        
        # Add any additional custom fields
        for key, value in kwargs.items():
            if key not in ["confirmation_candle", "trend_context"]:
                pattern[key] = value
        
        return pattern


class MarketConditionSignals:
    """Predefined signal sets for different market conditions."""
    
    @staticmethod
    def trending_market(
        symbol: str,
        direction: str = "up",
        base_price: float = 100.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Create a set of signals for a trending market.
        
        Args:
            symbol: Trading symbol
            direction: Trend direction ("up" or "down")
            base_price: Base price for calculations
            
        Returns:
            List of signals
        """
        signals = []
        
        # Set directionality based on trend
        buy_action = "buy" if direction == "up" else "sell"
        sell_action = "sell" if direction == "up" else "buy"
        
        price_factor = 1.01 if direction == "up" else 0.99
        current_price = base_price
        
        # Generate multiple entry signals with trend
        for i in range(3):
            # Modify price according to trend
            current_price = current_price * price_factor
            
            # TradingView entry signal
            tv_signal = TestSignalFactory.tradingview_entry_signal(
                symbol=symbol,
                direction=buy_action,
                price=current_price,
                timeframe=kwargs.get("timeframe", "1h"),
                confidence=random.uniform(0.75, 0.95),
                market_condition="trending"
            )
            signals.append(("tradingview", tv_signal))
            
            # Pattern confirmation
            pattern_signal = TestSignalFactory.pattern_detection(
                symbol=symbol,
                pattern_type=random.choice(["pin_bar", "engulfing"]),
                direction="long" if buy_action == "buy" else "short",
                price=current_price,
                confidence=random.uniform(0.80, 0.90),
                timeframe=kwargs.get("timeframe", "1h"),
                trend_context="continuation"
            )
            signals.append(("pattern", pattern_signal))
            
            # Add some consolidation
            current_price = current_price * (1.005 if direction == "up" else 0.995)
        
        return signals
    
    @staticmethod
    def ranging_market(
        symbol: str,
        base_price: float = 100.0,
        range_percent: float = 0.03,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Create a set of signals for a ranging market.
        
        Args:
            symbol: Trading symbol
            base_price: Base price for calculations
            range_percent: Size of the range as a percentage
            
        Returns:
            List of signals
        """
        signals = []
        
        # Calculate range boundaries
        upper_boundary = base_price * (1 + range_percent)
        lower_boundary = base_price * (1 - range_percent)
        
        # Signals at lower boundary (buy)
        lower_price = lower_boundary * (1 + random.uniform(0, 0.005))
        
        tv_buy_signal = TestSignalFactory.tradingview_entry_signal(
            symbol=symbol,
            direction="buy",
            price=lower_price,
            timeframe=kwargs.get("timeframe", "1h"),
            confidence=random.uniform(0.60, 0.80),
            market_condition="ranging",
            range_boundary="lower"
        )
        signals.append(("tradingview", tv_buy_signal))
        
        pattern_buy = TestSignalFactory.pattern_detection(
            symbol=symbol,
            pattern_type=random.choice(["double_bottom", "pin_bar"]),
            direction="long",
            price=lower_price,
            confidence=random.uniform(0.75, 0.85),
            timeframe=kwargs.get("timeframe", "1h"),
        )
        signals.append(("pattern", pattern_buy))
        
        # Signals at upper boundary (sell)
        upper_price = upper_boundary * (1 - random.uniform(0, 0.005))
        
        tv_sell_signal = TestSignalFactory.tradingview_entry_signal(
            symbol=symbol,
            direction="sell",
            price=upper_price,
            timeframe=kwargs.get("timeframe", "1h"),
            confidence=random.uniform(0.60, 0.80),
            market_condition="ranging",
            range_boundary="upper"
        )
        signals.append(("tradingview", tv_sell_signal))
        
        pattern_sell = TestSignalFactory.pattern_detection(
            symbol=symbol,
            pattern_type=random.choice(["double_top", "pin_bar"]),
            direction="short",
            price=upper_price,
            confidence=random.uniform(0.75, 0.85),
            timeframe=kwargs.get("timeframe", "1h"),
        )
        signals.append(("pattern", pattern_sell))
        
        return signals
    
    @staticmethod
    def breakout_scenario(
        symbol: str,
        base_price: float = 100.0,
        breakout_percent: float = 0.05,
        direction: str = "up",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Create a set of signals for a breakout scenario.
        
        Args:
            symbol: Trading symbol
            base_price: Base price for calculations
            breakout_percent: Size of the breakout as a percentage
            direction: Breakout direction ("up" or "down")
            
        Returns:
            List of signals
        """
        signals = []
        
        # Calculate breakout price
        breakout_factor = 1 + breakout_percent if direction == "up" else 1 - breakout_percent
        breakout_price = base_price * breakout_factor
        
        # TradingView breakout signal
        buy_action = "buy" if direction == "up" else "sell"
        
        tv_signal = TestSignalFactory.tradingview_entry_signal(
            symbol=symbol,
            direction=buy_action,
            price=breakout_price,
            timeframe=kwargs.get("timeframe", "1h"),
            confidence=random.uniform(0.80, 0.95),
            market_condition="breakout",
            volume_increase=random.uniform(2.0, 4.0)
        )
        signals.append(("tradingview", tv_signal))
        
        # Pattern confirmation (usually engulfing on breakouts)
        pattern_signal = TestSignalFactory.pattern_detection(
            symbol=symbol,
            pattern_type="engulfing",
            direction="long" if direction == "up" else "short",
            price=breakout_price,
            confidence=random.uniform(0.85, 0.95),
            timeframe=kwargs.get("timeframe", "1h"),
            trend_context="reversal"
        )
        signals.append(("pattern", pattern_signal))
        
        # Finnhub volume spike
        finnhub_signal = TestSignalFactory.finnhub_trade(
            symbol=symbol,
            price=breakout_price,
            volume=random.randint(500, 1000),  # High volume for breakout
            conditions=["@", "T"]  # Trade conditions
        )
        signals.append(("finnhub", finnhub_signal))
        
        return signals
    
    @staticmethod
    def reversal_scenario(
        symbol: str,
        base_price: float = 100.0,
        prior_move_percent: float = 0.10,
        direction: str = "bullish",  # bullish = down to up, bearish = up to down
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Create a set of signals for a market reversal scenario.
        
        Args:
            symbol: Trading symbol
            base_price: Current price
            prior_move_percent: Size of the move prior to reversal
            direction: Reversal direction ("bullish" or "bearish")
            
        Returns:
            List of signals
        """
        signals = []
        
        # Calculate prior trend end price
        if direction == "bullish":
            # Price moved down, now reversing up
            prior_trend_price = base_price * (1 - prior_move_percent)
            buy_action = "buy"
            pattern_direction = "long"
        else:
            # Price moved up, now reversing down
            prior_trend_price = base_price * (1 + prior_move_percent)
            buy_action = "sell"
            pattern_direction = "short"
        
        # TradingView reversal signal
        tv_signal = TestSignalFactory.tradingview_entry_signal(
            symbol=symbol,
            direction=buy_action,
            price=prior_trend_price,
            timeframe=kwargs.get("timeframe", "1h"),
            confidence=random.uniform(0.75, 0.90),
            market_condition="reversal",
            oversold=direction == "bullish",
            overbought=direction == "bearish"
        )
        signals.append(("tradingview", tv_signal))
        
        # Strong pattern signal (common in reversals)
        pattern_signal = TestSignalFactory.pattern_detection(
            symbol=symbol,
            pattern_type=random.choice(["engulfing", "pin_bar"]),
            direction=pattern_direction,
            price=prior_trend_price,
            confidence=random.uniform(0.85, 0.95),
            timeframe=kwargs.get("timeframe", "1h"),
            trend_context="reversal",
            confirmation_candle=True
        )
        signals.append(("pattern", pattern_signal))
        
        # Additional confirmation from higher timeframe
        higher_tf_pattern = TestSignalFactory.pattern_detection(
            symbol=symbol,
            pattern_type=random.choice(["double_bottom" if direction == "bullish" else "double_top", "engulfing"]),
            direction=pattern_direction,
            price=prior_trend_price,
            confidence=random.uniform(0.80, 0.90),
            timeframe="4h",  # Higher timeframe
            trend_context="reversal"
        )
        signals.append(("pattern", higher_tf_pattern))
        
        return signals
    
    @staticmethod
    def volatility_expansion(
        symbol: str,
        base_price: float = 100.0,
        volatility_increase: float = 2.5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Create signals for a volatility expansion scenario.
        
        Args:
            symbol: Trading symbol
            base_price: Base price
            volatility_increase: Factor by which volatility increases
            
        Returns:
            List of signals
        """
        signals = []
        
        # In volatility expansion, we often see contradicting signals
        
        # Initial move up
        up_price = base_price * 1.02
        
        # TradingView signal on volatility increase
        tv_signal = TestSignalFactory.tradingview_entry_signal(
            symbol=symbol,
            direction=random.choice(["buy", "sell"]),  # Random direction in volatility
            price=up_price,
            timeframe=kwargs.get("timeframe", "1h"),
            confidence=random.uniform(0.60, 0.75),  # Lower confidence in volatility
            market_condition="volatile",
            atr_increase=volatility_increase
        )
        signals.append(("tradingview", tv_signal))
        
        # Finnhub showing high volume and volatility
        for i in range(3):
            # Create rapidly changing prices
            price_change = random.choice([1.01, 0.99])
            current_price = up_price * price_change
            
            finnhub_signal = TestSignalFactory.finnhub_trade(
                symbol=symbol,
                price=current_price,
                volume=random.randint(300, 800)
            )
            signals.append(("finnhub", finnhub_signal))
        
        # Pattern detection with low confidence (hard to detect in volatility)
        pattern_signal = TestSignalFactory.pattern_detection(
            symbol=symbol,
            pattern_type=random.choice(["pin_bar", "engulfing"]),
            direction=random.choice(["long", "short"]),
            price=up_price * random.choice([1.01, 0.99]),
            confidence=random.uniform(0.50, 0.70),  # Lower confidence
            timeframe=kwargs.get("timeframe", "1h")
        )
        signals.append(("pattern", pattern_signal))
        
        return signals


class AssetClassSignals:
    """Signal sets for different asset classes with their specific characteristics."""
    
    @staticmethod
    def forex_signals(
        pair: str = "EURUSD",
        base_price: Optional[float] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Create forex-specific signals.
        
        Args:
            pair: Forex pair
            base_price: Base price (or use realistic defaults)
            
        Returns:
            List of signals
        """
        # Set realistic default prices for major pairs
        if base_price is None:
            defaults = {
                "EURUSD": 1.05,
                "GBPUSD": 1.27,
                "USDJPY": 150.0,
                "AUDUSD": 0.65,
                "USDCAD": 1.38
            }
            base_price = defaults.get(pair, 1.0)
        
        # Generate signals using appropriate pip sizes and behavior
        # For forex, we'll create signals for ranging and trending scenarios
        
        signals = []
        
        # Add ranging market signals (common in forex)
        if random.choice([True, False]):
            range_signals = MarketConditionSignals.ranging_market(
                symbol=pair,
                base_price=base_price,
                range_percent=0.005  # Small ranges in forex
            )
            signals.extend(range_signals)
        else:
            # Trend signals
            trend_signals = MarketConditionSignals.trending_market(
                symbol=pair,
                direction=random.choice(["up", "down"]),
                base_price=base_price
            )
            signals.extend(trend_signals)
        
        # Add forex-specific metadata
        for i, (source, signal) in enumerate(signals):
            if source == "tradingview":
                signal["asset_class"] = "forex"
                signal["lot_size"] = random.choice([0.01, 0.05, 0.1])
                signal["spread"] = random.uniform(0.5, 2.0)  # pips
                
                # Add appropriate stop loss and take profit for forex
                if "action" in signal and signal["action"] in ["buy", "sell"]:
                    direction_mult = 1 if signal["action"] == "buy" else -1
                    signal["stop_loss"] = base_price - (0.0020 * direction_mult)
                    signal["take_profit"] = base_price + (0.0050 * direction_mult)
            
        return signals
    
    @staticmethod
    def crypto_signals(
        symbol: str = "BTCUSD",
        base_price: Optional[float] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Create cryptocurrency-specific signals.
        
        Args:
            symbol: Crypto symbol
            base_price: Base price (or use realistic defaults)
            
        Returns:
            List of signals
        """
        # Set realistic default prices
        if base_price is None:
            defaults = {
                "BTCUSD": 35000,
                "ETHUSD": 1800,
                "SOLUSD": 75,
                "XRPUSD": 0.5
            }
            base_price = defaults.get(symbol, 100.0)
        
        signals = []
        
        # Crypto often shows breakouts and high volatility
        if random.choice([True, False]):
            breakout_signals = MarketConditionSignals.breakout_scenario(
                symbol=symbol,
                base_price=base_price,
                breakout_percent=0.08,  # Higher breakouts in crypto
                direction=random.choice(["up", "down"])
            )
            signals.extend(breakout_signals)
        else:
            # High volatility
            volatility_signals = MarketConditionSignals.volatility_expansion(
                symbol=symbol,
                base_price=base_price,
                volatility_increase=3.0  # Higher volatility in crypto
            )
            signals.extend(volatility_signals)
        
        # Add crypto-specific metadata
        for i, (source, signal) in enumerate(signals):
            if source == "tradingview":
                signal["asset_class"] = "crypto"
                signal["volume_btc"] = random.uniform(10, 100)
                signal["exchange"] = random.choice(["Binance", "Coinbase", "FTX"])
                
                # Add appropriate stop loss and take profit for crypto
                if "action" in signal and signal["action"] in ["buy", "sell"]:
                    direction_mult = 1 if signal["action"] == "buy" else -1
                    signal["stop_loss"] = base_price * (1 - (0.05 * direction_mult))
                    signal["take_profit"] = base_price * (1 + (0.15 * direction_mult))
            
        return signals
    
    @staticmethod
    def stock_signals(
        symbol: str = "AAPL",
        base_price: Optional[float] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Create stock-specific signals.
        
        Args:
            symbol: Stock symbol
            base_price: Base price (or use realistic defaults)
            
        Returns:
            List of signals
        """
        # Set realistic default prices
        if base_price is None:
            defaults = {
                "AAPL": 175,
                "MSFT": 350,
                "AMZN": 120,
                "GOOGL": 140,
                "META": 450
            }
            base_price = defaults.get(symbol, 100.0)
        
        signals = []
        
        # Stocks often respond to fundamentals and news
        # We'll simulate signals before/after earnings
        
        # Create a news-driven reversal
        reversal_signals = MarketConditionSignals.reversal_scenario(
            symbol=symbol,
            base_price=base_price,
            prior_move_percent=0.05,
            direction=random.choice(["bullish", "bearish"])
        )
        signals.extend(reversal_signals)
        
        # Add stock-specific metadata and Alpaca signals
        for i, (source, signal) in enumerate(signals):
            if source == "tradingview":
                signal["asset_class"] = "stock"
                signal["market_cap"] = random.uniform(100, 2000)  # billions
                signal["sector"] = random.choice(["Technology", "Consumer", "Finance", "Healthcare"])
                signal["earnings_proximity"] = random.choice(["pre_earnings", "post_earnings", "non_earnings"])
                
                # Add appropriate stop loss and take profit for stocks
                if "action" in signal and signal["action"] in ["buy", "sell"]:
                    direction_mult = 1 if signal["action"] == "buy" else -1
                    signal["stop_loss"] = base_price * (1 - (0.03 * direction_mult))
                    signal["take_profit"] = base_price * (1 + (0.08 * direction_mult))
        
        # Add an Alpaca order signal
        alpaca_signal = TestSignalFactory.alpaca_order_filled(
            symbol=symbol,
            side="buy" if random.random() > 0.5 else "sell",
            price=base_price * random.uniform(0.98, 1.02),
            quantity=random.randint(5, 20)
        )
        signals.append(("alpaca", alpaca_signal))
            
        return signals


def generate_multi_asset_test_set() -> List[Tuple[str, Dict[str, Any]]]:
    """
    Generate a comprehensive test set covering multiple assets and conditions.
    
    Returns:
        List of (source, signal) tuples
    """
    test_signals = []
    
    # Add forex signals
    forex_pairs = ["EURUSD", "GBPUSD", "USDJPY"]
    for pair in forex_pairs:
        signals = AssetClassSignals.forex_signals(pair=pair)
        test_signals.extend(signals)
    
    # Add crypto signals
    crypto_assets = ["BTCUSD", "ETHUSD"]
    for asset in crypto_assets:
        signals = AssetClassSignals.crypto_signals(symbol=asset)
        test_signals.extend(signals)
    
    # Add stock signals
    stocks = ["AAPL", "MSFT", "AMZN"]
    for stock in stocks:
        signals = AssetClassSignals.stock_signals(symbol=stock)
        test_signals.extend(signals)
    
    return test_signals


if __name__ == "__main__":
    # Generate and print some example signals
    print("Example TradingView Entry Signal:")
    print(json.dumps(TestSignalFactory.tradingview_entry_signal("EURUSD", "buy", 1.05), indent=2))
    
    print("\nExample Pattern Detection:")
    print(json.dumps(TestSignalFactory.pattern_detection("GBPUSD", "engulfing", "long", 1.25), indent=2))
    
    print("\nExample Trending Market Signals:")
    trend_signals = MarketConditionSignals.trending_market("USDJPY", "up", 150.0)
    for source, signal in trend_signals:
        print(f"\n{source.upper()} Signal:")
        print(json.dumps(signal, indent=2))
    
    print("\nGenerating multi-asset test set...")
    test_set = generate_multi_asset_test_set()
    print(f"Generated {len(test_set)} test signals across multiple assets")
