#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Pattern Recognition Test

This simplified version demonstrates pattern recognition capabilities
without depending on the existing trading_bot infrastructure.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PatternTest")

# Define enums for pattern structure
class PatternType(Enum):
    PRICE_ACTION = "price_action"
    CANDLESTICK = "candlestick"
    CHART_FORMATION = "chart_formation"
    INDICATOR_SIGNAL = "indicator_signal"
    VOLATILITY_BASED = "volatility_based"

class MarketContext(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class Pattern:
    """Base class for trading patterns"""
    
    def __init__(self, name, pattern_type, description=""):
        self.name = name
        self.pattern_type = pattern_type
        self.description = description
        self.success_rate = 0.0
        self.occurrences = 0
        self.best_context = {}
        self.parameters = {}

class PriceActionDetector:
    """Detects price action patterns like pin bars and inside/outside bars"""
    
    def detect_pattern(self, data, pattern_name):
        """Detect a specific price action pattern"""
        if len(data) < 3:  # Need at least 3 bars
            return {"match": False, "confidence": 0.0}
        
        # Get the most recent bars
        recent_data = data.iloc[-3:]
        current_bar = recent_data.iloc[-1]
        previous_bar = recent_data.iloc[-2]
        
        result = {
            "match": False,
            "confidence": 0.0,
            "direction": None,
            "pattern_name": pattern_name
        }
        
        # Detect pin bar
        if pattern_name == "pin_bar":
            # Calculate body and wick sizes
            body_size = abs(current_bar['Close'] - current_bar['Open'])
            total_range = current_bar['High'] - current_bar['Low']
            
            if total_range == 0:  # Avoid division by zero
                return result
            
            body_ratio = body_size / total_range
            
            # Calculate wick sizes
            if current_bar['Close'] >= current_bar['Open']:  # Bullish candle
                upper_wick = current_bar['High'] - current_bar['Close']
                lower_wick = current_bar['Open'] - current_bar['Low']
            else:  # Bearish candle
                upper_wick = current_bar['High'] - current_bar['Open']
                lower_wick = current_bar['Close'] - current_bar['Low']
            
            upper_wick_ratio = upper_wick / total_range
            lower_wick_ratio = lower_wick / total_range
            
            # Pin bar criteria: small body, one long wick
            is_pin_bar = (
                body_ratio <= 0.3 and  # Small body
                (
                    (upper_wick_ratio >= 0.6 and lower_wick_ratio <= 0.2) or  # Long upper wick
                    (lower_wick_ratio >= 0.6 and upper_wick_ratio <= 0.2)     # Long lower wick
                )
            )
            
            if is_pin_bar:
                # Determine direction
                if upper_wick_ratio > lower_wick_ratio:
                    result["direction"] = "sell"  # Upper wick pin bar is bearish
                else:
                    result["direction"] = "buy"   # Lower wick pin bar is bullish
                
                # Calculate confidence based on wick/body ratio
                confidence = min(0.95, max(0.6, 1 - body_ratio))
                
                result["match"] = True
                result["confidence"] = confidence
        
        # Detect inside bar
        elif pattern_name == "inside_bar":
            # Inside bar criteria
            is_inside = (
                current_bar['High'] <= previous_bar['High'] and
                current_bar['Low'] >= previous_bar['Low']
            )
            
            if is_inside:
                # Calculate how "inside" the bar is (higher = better)
                current_range = current_bar['High'] - current_bar['Low']
                previous_range = previous_bar['High'] - previous_bar['Low']
                
                if previous_range > 0:
                    inside_ratio = 1 - (current_range / previous_range)
                    confidence = min(0.9, 0.6 + (inside_ratio * 0.3))
                else:
                    confidence = 0.6
                
                # Direction based on previous bar
                if previous_bar['Close'] > previous_bar['Open']:
                    result["direction"] = "buy"   # Previous bar bullish
                else:
                    result["direction"] = "sell"  # Previous bar bearish
                
                result["match"] = True
                result["confidence"] = confidence
        
        # Detect outside bar
        elif pattern_name == "outside_bar":
            # Outside bar criteria
            is_outside = (
                current_bar['High'] > previous_bar['High'] and
                current_bar['Low'] < previous_bar['Low']
            )
            
            if is_outside:
                # Direction based on current bar
                if current_bar['Close'] > current_bar['Open']:
                    result["direction"] = "buy"
                else:
                    result["direction"] = "sell"
                
                # Confidence based on how much it engulfs
                confidence = 0.75
                
                result["match"] = True
                result["confidence"] = confidence
        
        return result

class CandlestickDetector:
    """Detects candlestick patterns like engulfing, doji, etc."""
    
    def detect_pattern(self, data, pattern_name):
        """Detect a specific candlestick pattern"""
        if len(data) < 3:  # Need at least 3 bars
            return {"match": False, "confidence": 0.0}
        
        # Get the most recent bars
        recent_data = data.iloc[-3:]
        current_bar = recent_data.iloc[-1]
        previous_bar = recent_data.iloc[-2]
        
        result = {
            "match": False,
            "confidence": 0.0,
            "direction": None,
            "pattern_name": pattern_name
        }
        
        # Detect engulfing pattern
        if pattern_name == "engulfing":
            # Calculate bodies
            current_high = max(current_bar['Open'], current_bar['Close'])
            current_low = min(current_bar['Open'], current_bar['Close'])
            
            previous_high = max(previous_bar['Open'], previous_bar['Close'])
            previous_low = min(previous_bar['Open'], previous_bar['Close'])
            
            # Engulfing criteria
            is_engulfing = (current_high > previous_high and current_low < previous_low)
            
            # Check if current and previous bars have opposite directions
            current_bullish = current_bar['Close'] > current_bar['Open']
            previous_bullish = previous_bar['Close'] > previous_bar['Open']
            
            if is_engulfing and current_bullish != previous_bullish:
                result["match"] = True
                result["direction"] = "buy" if current_bullish else "sell"
                result["confidence"] = 0.75
        
        # Detect doji
        elif pattern_name == "doji":
            # Calculate body and total range
            body_size = abs(current_bar['Close'] - current_bar['Open'])
            total_range = current_bar['High'] - current_bar['Low']
            
            if total_range > 0:
                # Doji has very small body compared to range
                body_ratio = body_size / total_range
                is_doji = body_ratio < 0.1
                
                if is_doji:
                    result["match"] = True
                    # Doji doesn't have a clear direction on its own
                    result["direction"] = None
                    result["confidence"] = 0.7
        
        # Detect morning star (bullish reversal)
        elif pattern_name == "morning_star":
            if len(data) < 4:  # Need at least 4 bars
                return result
                
            # Get three bars for the pattern
            first_bar = recent_data.iloc[-3]
            middle_bar = recent_data.iloc[-2]
            last_bar = recent_data.iloc[-1]
            
            # Morning star criteria:
            # 1. First bar is bearish with large body
            # 2. Second bar has small body (doji-like)
            # 3. Third bar is bullish with large body
            
            first_bearish = first_bar['Close'] < first_bar['Open']
            first_body = abs(first_bar['Close'] - first_bar['Open'])
            
            middle_body = abs(middle_bar['Close'] - middle_bar['Open'])
            middle_range = middle_bar['High'] - middle_bar['Low']
            small_middle = middle_body / middle_range < 0.3 if middle_range > 0 else False
            
            last_bullish = last_bar['Close'] > last_bar['Open']
            last_body = abs(last_bar['Close'] - last_bar['Open'])
            
            # Check if pattern forms
            if first_bearish and small_middle and last_bullish and last_body > 0.5 * first_body:
                result["match"] = True
                result["direction"] = "buy"
                result["confidence"] = 0.8
        
        return result

class PatternRecognitionTest:
    """
    Tests pattern recognition capabilities on synthetic or real market data.
    """
    
    def __init__(self, data_dir="data/test_data"):
        """Initialize the pattern tester"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize pattern detectors
        self.price_action_detector = PriceActionDetector()
        self.candlestick_detector = CandlestickDetector()
        
        # Pattern statistics
        self.stats = {
            "total_bars": 0,
            "patterns_detected": 0,
            "by_regime": {},
            "by_pattern": {}
        }
    
    def generate_test_data(self, symbol="TEST", days=100):
        """
        Generate synthetic test data with embedded patterns.
        
        Args:
            symbol: Symbol name for the test data
            days: Number of days to generate
            
        Returns:
            DataFrame with test data
        """
        logger.info(f"Generating {days} days of test data for {symbol}")
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Initial price
        price = 100.0
        
        # Data lists
        data = []
        
        # Regimes we'll cycle through
        regimes = [
            "trending_up", "volatile", "ranging", 
            "trending_down", "breakout", "ranging"
        ]
        
        # Generate price series with different regimes
        regime_idx = 0
        regime_duration = days // len(regimes)
        regime_days_left = regime_duration
        current_regime = regimes[regime_idx]
        
        # Trend direction and volatility state
        trend_direction = 1  # 1 for up, -1 for down
        volatility = 0.01    # Base volatility
        
        for i, date in enumerate(dates):
            # Check if we need to switch regimes
            if regime_days_left <= 0:
                regime_idx = (regime_idx + 1) % len(regimes)
                current_regime = regimes[regime_idx]
                regime_days_left = regime_duration
                
                logger.info(f"Switching to {current_regime} regime at day {i}")
                
                # Adjust parameters for the new regime
                if current_regime == "trending_up":
                    trend_direction = 1
                    volatility = 0.01
                elif current_regime == "trending_down":
                    trend_direction = -1
                    volatility = 0.01
                elif current_regime == "ranging":
                    trend_direction = 0
                    volatility = 0.005
                elif current_regime == "volatile":
                    trend_direction = trend_direction  # Keep current trend
                    volatility = 0.025
                elif current_regime == "breakout":
                    # Reverse trend direction for breakout
                    trend_direction = -trend_direction
                    volatility = 0.02
            
            # Generate price move based on regime
            base_move = 0
            
            # Add trend component
            if trend_direction != 0:
                base_move += trend_direction * 0.0005
            
            # Add random component
            random_move = np.random.normal(0, volatility)
            daily_return = base_move + random_move
            
            # Update price
            price *= (1 + daily_return)
            
            # Calculate high, low, open prices
            daily_range = price * volatility * 2
            high = price + daily_range / 2
            low = price - daily_range / 2
            open_price = low + np.random.uniform(0, daily_range)
            
            # Occasionally insert known patterns
            has_pattern = False
            pattern_name = None
            
            # Every ~10 days, insert a pattern
            if i % 10 == 0:
                # Alternate different pattern types
                pattern_idx = (i // 10) % 3
                
                if pattern_idx == 0 and current_regime in ["trending_up", "trending_down", "reversal"]:
                    # Create a pin bar
                    has_pattern = True
                    pattern_name = "pin_bar"
                    
                    if current_regime == "trending_down":
                        # Bullish pin bar (long lower wick)
                        low = price - daily_range * 2
                        high = price + daily_range * 0.3
                        close_price = price + daily_range * 0.1
                        open_price = price
                    else:
                        # Bearish pin bar (long upper wick)
                        high = price + daily_range * 2
                        low = price - daily_range * 0.3
                        close_price = price - daily_range * 0.1
                        open_price = price
                
                elif pattern_idx == 1 and current_regime in ["trending_up", "trending_down"]:
                    # Create an engulfing pattern (need previous bar)
                    if i > 0:
                        has_pattern = True
                        pattern_name = "engulfing"
                        
                        prev_data = data[-1]
                        
                        if current_regime == "trending_down":
                            # Bullish engulfing
                            open_price = prev_data["Close"] * 0.99
                            close_price = prev_data["Open"] * 1.02
                            low = open_price * 0.99
                            high = close_price * 1.01
                        else:
                            # Bearish engulfing
                            open_price = prev_data["Close"] * 1.01
                            close_price = prev_data["Open"] * 0.98
                            low = close_price * 0.99
                            high = open_price * 1.01
                
                elif pattern_idx == 2 and current_regime == "ranging":
                    # Create a doji
                    has_pattern = True
                    pattern_name = "doji"
                    
                    # Tiny body, equal wicks
                    mid_price = (price + open_price) / 2
                    open_price = mid_price * 0.999
                    close_price = mid_price * 1.001
                    high = mid_price * 1.015
                    low = mid_price * 0.985
            
            # Add to data list
            data.append({
                "Date": date,
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": price if not has_pattern else close_price,
                "Volume": np.random.randint(1000, 10000),
                "Regime": current_regime,
                "HasPattern": has_pattern,
                "PatternName": pattern_name
            })
            
            # Decrement regime counter
            regime_days_left -= 1
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        
        # Calculate technical indicators
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Save to CSV
        csv_path = os.path.join(self.data_dir, f"{symbol}_test.csv")
        df.to_csv(csv_path)
        logger.info(f"Saved test data to {csv_path}")
        
        return df
    
    def detect_patterns(self, data, min_confidence=0.6):
        """
        Detect patterns in the provided market data.
        
        Args:
            data: Market data as DataFrame
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detected patterns
        """
        # Patterns to detect
        patterns_to_check = [
            {"name": "pin_bar", "type": PatternType.PRICE_ACTION},
            {"name": "engulfing", "type": PatternType.CANDLESTICK},
            {"name": "doji", "type": PatternType.CANDLESTICK},
            {"name": "inside_bar", "type": PatternType.PRICE_ACTION},
            {"name": "morning_star", "type": PatternType.CANDLESTICK}
        ]
        
        # Results storage
        detected_patterns = []
        
        # Check each pattern
        for pattern in patterns_to_check:
            result = None
            
            if pattern["type"] == PatternType.PRICE_ACTION:
                result = self.price_action_detector.detect_pattern(data, pattern["name"])
            elif pattern["type"] == PatternType.CANDLESTICK:
                result = self.candlestick_detector.detect_pattern(data, pattern["name"])
            
            if result and result["match"] and result["confidence"] >= min_confidence:
                detected_patterns.append(result)
                
                # Update statistics
                self.stats["patterns_detected"] += 1
                
                # Track by pattern
                if pattern["name"] not in self.stats["by_pattern"]:
                    self.stats["by_pattern"][pattern["name"]] = 0
                self.stats["by_pattern"][pattern["name"]] += 1
                
                # Track by regime if available
                if "Regime" in data.iloc[-1]:
                    regime = data.iloc[-1]["Regime"]
                    if regime not in self.stats["by_regime"]:
                        self.stats["by_regime"][regime] = 0
                    self.stats["by_regime"][regime] += 1
        
        # Sort by confidence
        detected_patterns.sort(key=lambda x: x["confidence"], reverse=True)
        
        return detected_patterns
    
    def run_pattern_test(self, data=None, symbol="TEST"):
        """
        Run a complete pattern recognition test.
        
        Args:
            data: Market data or None to generate synthetic data
            symbol: Symbol to use if generating data
            
        Returns:
            Dictionary with test results
        """
        # Generate data if not provided
        if data is None:
            data = self.generate_test_data(symbol)
        
        logger.info(f"Running pattern recognition test on {len(data)} bars")
        
        # Results storage
        results = {
            "total_bars": len(data),
            "detected_patterns": [],
            "detection_by_regime": {},
            "signals": []
        }
        
        # Window size for detection
        window_size = 20
        
        # Analyze each bar with a rolling window
        for i in range(window_size, len(data)):
            # Get data window
            window = data.iloc[i-window_size:i+1]
            
            # Detect patterns
            patterns = self.detect_patterns(window)
            
            if patterns:
                # Get current regime if available
                regime = "unknown"
                if "Regime" in data.iloc[i]:
                    regime = data.iloc[i]["Regime"]
                
                # Store detection
                detection = {
                    "date": data.index[i],
                    "bar_idx": i,
                    "regime": regime,
                    "patterns": patterns
                }
                
                results["detected_patterns"].append(detection)
                
                # Track by regime
                if regime not in results["detection_by_regime"]:
                    results["detection_by_regime"][regime] = []
                results["detection_by_regime"][regime].append(detection)
                
                # Generate trade signal from best pattern
                best_pattern = patterns[0]
                if best_pattern["direction"]:
                    signal = {
                        "date": data.index[i],
                        "bar_idx": i,
                        "signal": best_pattern["direction"],
                        "pattern": best_pattern["pattern_name"],
                        "confidence": best_pattern["confidence"],
                        "regime": regime
                    }
                    
                    results["signals"].append(signal)
        
        # Update statistics
        self.stats["total_bars"] += len(data)
        
        logger.info(f"Pattern recognition results:")
        logger.info(f"  Total bars analyzed: {len(data)}")
        logger.info(f"  Patterns detected: {len(results['detected_patterns'])}")
        logger.info(f"  Trade signals generated: {len(results['signals'])}")
        
        # Pattern detection rate
        if len(data) > 0:
            detection_rate = len(results['detected_patterns']) / len(data)
            logger.info(f"  Pattern detection rate: {detection_rate:.2%}")
        
        # By regime
        logger.info("  Detection by regime:")
        for regime, detections in results["detection_by_regime"].items():
            logger.info(f"    {regime}: {len(detections)} patterns")
        
        return results
    
    def print_statistics(self):
        """Print overall pattern recognition statistics"""
        logger.info("\n=== PATTERN RECOGNITION STATISTICS ===")
        logger.info(f"Total bars analyzed: {self.stats['total_bars']}")
        logger.info(f"Total patterns detected: {self.stats['patterns_detected']}")
        
        # Pattern detection rate
        if self.stats['total_bars'] > 0:
            detection_rate = self.stats['patterns_detected'] / self.stats['total_bars']
            logger.info(f"Pattern detection rate: {detection_rate:.2%}")
        
        # By pattern type
        logger.info("\nDetection by pattern:")
        for pattern, count in self.stats['by_pattern'].items():
            logger.info(f"  {pattern}: {count}")
        
        # By regime
        logger.info("\nDetection by regime:")
        for regime, count in self.stats['by_regime'].items():
            logger.info(f"  {regime}: {count}")

def main():
    """Main function to run the pattern recognition test"""
    # Create tester
    tester = PatternRecognitionTest()
    
    # Run test with synthetic data
    results = tester.run_pattern_test()
    
    # Print overall statistics
    tester.print_statistics()
    
    return results

if __name__ == "__main__":
    main()
