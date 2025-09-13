#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fundamental Strategy - Buffett-style fundamental analysis strategies for trading
that integrate with the StrategyRotator system.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import json
from scipy import stats

# Import base strategy class
from trading_bot.strategy.strategy_rotator import Strategy
from trading_bot.common.config_utils import setup_directories, load_config, save_state, load_state

# Setup logging
logger = logging.getLogger("FundamentalStrategy")

class FundamentalStrategy(Strategy):
    """Base class for fundamental analysis trading strategies"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a fundamental strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        
        # Setup paths
        self.paths = setup_directories(
            data_dir=config.get("data_dir"),
            component_name=f"fundamental_strategy_{name}"
        )
        
        # Fundamental data cache
        self.fundamental_data = {}
        self.last_update_dates = {}
        
        # Default config for fundamental analysis
        self.valuation_weight = config.get("valuation_weight", 0.4)
        self.quality_weight = config.get("quality_weight", 0.3)
        self.financial_health_weight = config.get("financial_health_weight", 0.3)
        
        # Update frequency in seconds
        self.update_frequency = config.get("update_frequency", 86400)  # Daily by default
        
        # Load cached fundamental data if available
        self._load_cached_data()
    
    def _load_cached_data(self) -> None:
        """Load cached fundamental data from disk."""
        cache_path = os.path.join(self.paths["data_dir"], "fundamental_cache.json")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    self.fundamental_data = data.get("data", {})
                    self.last_update_dates = {
                        k: datetime.fromisoformat(v) if v else None
                        for k, v in data.get("last_updates", {}).items()
                    }
                logger.info(f"Loaded fundamental data cache for {len(self.fundamental_data)} symbols")
            except Exception as e:
                logger.error(f"Error loading fundamental data cache: {e}")
    
    def _save_cached_data(self) -> None:
        """Save fundamental data cache to disk."""
        cache_path = os.path.join(self.paths["data_dir"], "fundamental_cache.json")
        
        try:
            # Convert datetime objects to ISO format strings
            last_updates = {
                k: v.isoformat() if v else None
                for k, v in self.last_update_dates.items()
            }
            
            with open(cache_path, 'w') as f:
                json.dump({
                    "data": self.fundamental_data,
                    "last_updates": last_updates
                }, f)
            logger.info(f"Saved fundamental data cache for {len(self.fundamental_data)} symbols")
        except Exception as e:
            logger.error(f"Error saving fundamental data cache: {e}")
    
    def update_fundamental_data(self, symbols: List[str], force: bool = False) -> None:
        """
        Update fundamental data for specified symbols.
        
        Args:
            symbols: List of symbols to update
            force: Force update even if data is recent
        """
        now = datetime.now()
        updated_symbols = []
        
        for symbol in symbols:
            # Check if update is needed
            last_update = self.last_update_dates.get(symbol)
            
            if (not force and last_update and 
                (now - last_update).total_seconds() < self.update_frequency):
                logger.debug(f"Skipping update for {symbol}, data is recent")
                continue
            
            try:
                # Fetch data for symbol
                data = self._fetch_fundamental_data(symbol)
                
                if data:
                    self.fundamental_data[symbol] = data
                    self.last_update_dates[symbol] = now
                    updated_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Error updating fundamental data for {symbol}: {e}")
        
        if updated_symbols:
            logger.info(f"Updated fundamental data for {len(updated_symbols)} symbols")
            self._save_cached_data()
    
    def _fetch_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch fundamental data for a symbol.
        
        Args:
            symbol: Symbol to fetch data for
            
        Returns:
            Dict with fundamental data
        """
        # In a real implementation, this would fetch data from a provider
        # such as Alpha Vantage, Yahoo Finance, or a proprietary API
        
        logger.warning(f"Using mock fundamental data for {symbol}")
        
        # Mock data for demonstration purposes
        return {
            "valuation": {
                "pe_ratio": 15 + np.random.normal(0, 3),
                "price_to_book": 2.5 + np.random.normal(0, 0.5),
                "ev_to_ebitda": 10 + np.random.normal(0, 2),
                "price_to_fcf": 12 + np.random.normal(0, 3),
                "dividend_yield": 0.02 + np.random.normal(0, 0.005)
            },
            "quality": {
                "return_on_equity": 0.15 + np.random.normal(0, 0.03),
                "return_on_assets": 0.08 + np.random.normal(0, 0.02),
                "gross_margin": 0.4 + np.random.normal(0, 0.05),
                "operating_margin": 0.15 + np.random.normal(0, 0.03),
                "net_margin": 0.1 + np.random.normal(0, 0.02)
            },
            "financial_health": {
                "current_ratio": 1.5 + np.random.normal(0, 0.2),
                "debt_to_equity": 0.5 + np.random.normal(0, 0.1),
                "interest_coverage": 8 + np.random.normal(0, 2),
                "debt_to_ebitda": 2.5 + np.random.normal(0, 0.5),
                "free_cash_flow": 500e6 + np.random.normal(0, 100e6)
            },
            "growth": {
                "revenue_growth_3yr": 0.1 + np.random.normal(0, 0.02),
                "eps_growth_3yr": 0.12 + np.random.normal(0, 0.03),
                "fcf_growth_3yr": 0.08 + np.random.normal(0, 0.03)
            },
            "dcf": {
                "intrinsic_value": 100 + np.random.normal(0, 10),
                "current_price": 90 + np.random.normal(0, 8),
                "margin_of_safety": 0.1 + np.random.normal(0, 0.05)
            }
        }
    
    def calculate_valuation_score(self, symbol: str) -> float:
        """
        Calculate valuation score for a symbol.
        
        Args:
            symbol: Symbol to calculate score for
            
        Returns:
            float: Valuation score between -1.0 and 1.0
        """
        if symbol not in self.fundamental_data:
            logger.warning(f"No fundamental data for {symbol}")
            return 0.0
        
        data = self.fundamental_data[symbol]
        valuation = data.get("valuation", {})
        dcf = data.get("dcf", {})
        
        # Calculate DCF component (margin of safety)
        dcf_score = dcf.get("margin_of_safety", 0)
        
        # Calculate valuation ratios component
        # Lower is better for these ratios, so we invert
        pe_score = -self._normalize_value(valuation.get("pe_ratio", 15), 5, 25)
        pb_score = -self._normalize_value(valuation.get("price_to_book", 2.5), 1, 4)
        evebitda_score = -self._normalize_value(valuation.get("ev_to_ebitda", 10), 5, 15)
        pcf_score = -self._normalize_value(valuation.get("price_to_fcf", 12), 5, 20)
        
        # Higher is better for dividend yield
        div_score = self._normalize_value(valuation.get("dividend_yield", 0.02), 0, 0.05)
        
        # Combine scores (with equal weights for simplicity)
        ratio_score = np.mean([pe_score, pb_score, evebitda_score, pcf_score, div_score])
        
        # Combine DCF and ratio components
        valuation_score = 0.6 * dcf_score + 0.4 * ratio_score
        
        # Ensure score is between -1 and 1
        return np.clip(valuation_score, -1.0, 1.0)
    
    def calculate_quality_score(self, symbol: str) -> float:
        """
        Calculate quality score for a symbol.
        
        Args:
            symbol: Symbol to calculate score for
            
        Returns:
            float: Quality score between -1.0 and 1.0
        """
        if symbol not in self.fundamental_data:
            logger.warning(f"No fundamental data for {symbol}")
            return 0.0
        
        data = self.fundamental_data[symbol]
        quality = data.get("quality", {})
        
        # Higher is better for all these metrics
        roe_score = self._normalize_value(quality.get("return_on_equity", 0.15), 0, 0.3)
        roa_score = self._normalize_value(quality.get("return_on_assets", 0.08), 0, 0.15)
        gm_score = self._normalize_value(quality.get("gross_margin", 0.4), 0.2, 0.6)
        om_score = self._normalize_value(quality.get("operating_margin", 0.15), 0.05, 0.25)
        nm_score = self._normalize_value(quality.get("net_margin", 0.1), 0.03, 0.2)
        
        # Combine scores (with equal weights for simplicity)
        quality_score = np.mean([roe_score, roa_score, gm_score, om_score, nm_score])
        
        # Ensure score is between -1 and 1
        return np.clip(quality_score, -1.0, 1.0)
    
    def calculate_financial_health_score(self, symbol: str) -> float:
        """
        Calculate financial health score for a symbol.
        
        Args:
            symbol: Symbol to calculate score for
            
        Returns:
            float: Financial health score between -1.0 and 1.0
        """
        if symbol not in self.fundamental_data:
            logger.warning(f"No fundamental data for {symbol}")
            return 0.0
        
        data = self.fundamental_data[symbol]
        health = data.get("financial_health", {})
        
        # Higher is better for these
        cr_score = self._normalize_value(health.get("current_ratio", 1.5), 1, 3)
        ic_score = self._normalize_value(health.get("interest_coverage", 8), 2, 15)
        
        # Lower is better for these
        de_score = -self._normalize_value(health.get("debt_to_equity", 0.5), 0, 2)
        debt_ebitda_score = -self._normalize_value(health.get("debt_to_ebitda", 2.5), 0, 5)
        
        # Combine scores (with equal weights for simplicity)
        health_score = np.mean([cr_score, ic_score, de_score, debt_ebitda_score])
        
        # Ensure score is between -1 and 1
        return np.clip(health_score, -1.0, 1.0)
    
    def _normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize a value to the range [-1, 1].
        
        Args:
            value: Value to normalize
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            float: Normalized value between -1 and 1
        """
        range_val = max_val - min_val
        if range_val == 0:
            return 0.0
        
        # Map to [0, 1]
        normalized = (value - min_val) / range_val
        
        # Map to [-1, 1]
        return 2 * normalized - 1
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate a trading signal based on fundamental analysis.
        
        Args:
            market_data: Current market data
            
        Returns:
            float: Signal between -1.0 (strong sell) and 1.0 (strong buy)
        """
        symbol = market_data.get("symbol", "UNKNOWN")
        
        # Update fundamental data if needed
        if symbol not in self.fundamental_data:
            self.update_fundamental_data([symbol])
        
        # Calculate component scores
        valuation_score = self.calculate_valuation_score(symbol)
        quality_score = self.calculate_quality_score(symbol)
        health_score = self.calculate_financial_health_score(symbol)
        
        # Generate weighted signal
        signal = (
            valuation_score * self.valuation_weight +
            quality_score * self.quality_weight +
            health_score * self.financial_health_weight
        )
        
        # Update last signal and time
        self.last_signal = signal
        self.last_update_time = datetime.now()
        
        logger.debug(f"Generated fundamental signal for {symbol}: {signal:.4f}")
        logger.debug(f"Components: V={valuation_score:.2f}, Q={quality_score:.2f}, H={health_score:.2f}")
        
        return signal
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary for serialization."""
        base_dict = super().to_dict()
        # Add any additional fields specific to fundamental strategies
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FundamentalStrategy':
        """Create strategy from dictionary."""
        return super().from_dict(data)


class DCFStrategy(FundamentalStrategy):
    """
    Discounted Cash Flow (DCF) based fundamental strategy.
    Focuses on intrinsic value and margin of safety.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize DCF strategy"""
        super().__init__(name, config)
        
        # DCF-specific parameters
        self.required_margin = config.get("required_margin_of_safety", 0.15)
        self.max_signal_at_margin = config.get("max_signal_at_margin", 0.5)
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate a trading signal based on DCF analysis.
        
        Args:
            market_data: Current market data
            
        Returns:
            float: Signal between -1.0 (strong sell) and 1.0 (strong buy)
        """
        symbol = market_data.get("symbol", "UNKNOWN")
        
        # Update fundamental data if needed
        if symbol not in self.fundamental_data:
            self.update_fundamental_data([symbol])
        
        # Get DCF data
        if symbol not in self.fundamental_data:
            logger.warning(f"No fundamental data for {symbol}")
            return 0.0
        
        dcf_data = self.fundamental_data[symbol].get("dcf", {})
        
        # Calculate margin of safety
        intrinsic_value = dcf_data.get("intrinsic_value", 0)
        current_price = dcf_data.get("current_price", 0)
        
        if intrinsic_value <= 0 or current_price <= 0:
            logger.warning(f"Invalid price data for {symbol}")
            return 0.0
        
        margin_of_safety = (intrinsic_value - current_price) / intrinsic_value
        
        # Convert margin of safety to signal
        if margin_of_safety < 0:
            # Overvalued - negative signal
            signal = -min(abs(margin_of_safety) / self.max_signal_at_margin, 1.0)
        else:
            # Undervalued - positive signal
            signal = min(margin_of_safety / self.max_signal_at_margin, 1.0)
        
        # Update last signal and time
        self.last_signal = signal
        self.last_update_time = datetime.now()
        
        logger.debug(f"Generated DCF signal for {symbol}: {signal:.4f} (MoS: {margin_of_safety:.2%})")
        
        return signal


class QualityValueStrategy(FundamentalStrategy):
    """
    Quality-Value strategy that focuses on companies with strong fundamentals
    that are also reasonably priced.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize Quality-Value strategy"""
        super().__init__(name, config)
        
        # QV-specific parameters
        self.quality_threshold = config.get("quality_threshold", 0.3)
        self.value_weight = config.get("value_weight", 0.5)
        self.quality_weight = config.get("quality_weight", 0.5)
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate a trading signal based on quality and value.
        
        Args:
            market_data: Current market data
            
        Returns:
            float: Signal between -1.0 (strong sell) and 1.0 (strong buy)
        """
        symbol = market_data.get("symbol", "UNKNOWN")
        
        # Update fundamental data if needed
        if symbol not in self.fundamental_data:
            self.update_fundamental_data([symbol])
        
        # Calculate valuation and quality scores
        valuation_score = self.calculate_valuation_score(symbol)
        quality_score = self.calculate_quality_score(symbol)
        
        # Only generate positive signals for high-quality companies
        if quality_score < self.quality_threshold:
            signal = -0.5  # Negative signal for low-quality companies
        else:
            # For high-quality companies, weight quality and value
            signal = (
                valuation_score * self.value_weight + 
                quality_score * self.quality_weight
            )
        
        # Update last signal and time
        self.last_signal = signal
        self.last_update_time = datetime.now()
        
        logger.debug(f"Generated QV signal for {symbol}: {signal:.4f}")
        logger.debug(f"Components: V={valuation_score:.2f}, Q={quality_score:.2f}")
        
        return signal


class FinancialHealthStrategy(FundamentalStrategy):
    """
    Strategy focusing on financial health metrics like balance sheet strength
    and debt levels.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize Financial Health strategy"""
        super().__init__(name, config)
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate a trading signal based on financial health.
        
        Args:
            market_data: Current market data
            
        Returns:
            float: Signal between -1.0 (strong sell) and 1.0 (strong buy)
        """
        symbol = market_data.get("symbol", "UNKNOWN")
        
        # Update fundamental data if needed
        if symbol not in self.fundamental_data:
            self.update_fundamental_data([symbol])
        
        # Financial health is the primary focus
        health_score = self.calculate_financial_health_score(symbol)
        
        # Also consider quality as a secondary factor
        quality_score = self.calculate_quality_score(symbol)
        
        # Weight more heavily towards financial health
        signal = health_score * 0.8 + quality_score * 0.2
        
        # Update last signal and time
        self.last_signal = signal
        self.last_update_time = datetime.now()
        
        logger.debug(f"Generated Financial Health signal for {symbol}: {signal:.4f}")
        
        return signal


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a DCF strategy
    dcf_strategy = DCFStrategy("DCFStrategy")
    
    # Create mock market data
    market_data = {
        "symbol": "AAPL",
        "price": 150.0,
        "volume": 1000000
    }
    
    # Generate signal
    signal = dcf_strategy.generate_signal(market_data)
    print(f"DCF signal for AAPL: {signal:.4f}")
    
    # Try another strategy
    qv_strategy = QualityValueStrategy("QualityValueStrategy")
    signal = qv_strategy.generate_signal(market_data)
    print(f"Quality-Value signal for AAPL: {signal:.4f}") 