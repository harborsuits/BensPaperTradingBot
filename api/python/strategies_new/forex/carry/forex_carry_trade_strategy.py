"""
Forex Carry Trade Strategy

This strategy capitalizes on interest rate differentials between currencies by buying 
high-interest currencies and selling low-interest currencies. It focuses on capturing 
the positive swap/rollover payments while managing market risk through trend, volatility,
and correlation filters.

Features:
- Dynamic interest rate differential tracking
- Volatility-adjusted position sizing
- Correlation-based portfolio construction
- Central bank policy cycle analysis
- Carry-to-risk ratio optimization
- Multi-timeframe confirmation
- Risk-adjusted return monitoring
"""

import logging
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Any, Optional, Set, Union

from trading_bot.strategies_new.forex.base.forex_base_strategy import ForexBaseStrategy
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.models.signal import Signal


@register_strategy(
    asset_class="forex",
    strategy_type="carry",
    name="ForexCarryTrade",
    description="Capitalizes on interest rate differentials between currencies with volatility and trend filters",
    parameters={
        "default": {
            # Interest rate parameters
            "min_interest_rate_differential": 1.0,  # Minimum differential in percentage points
            "interest_rates_refresh_days": 7,  # How often to refresh interest rate data
            
            # Filter parameters
            "use_trend_filter": True,  # Whether to use trend filters
            "trend_filter_timeframes": ["1d", "1w"],  # Timeframes for trend confirmation
            "min_positive_timeframes": 1,  # Minimum number of timeframes that must show positive trend
            
            # Volatility parameters
            "max_volatility_atr": 1.5,  # Maximum ATR threshold (% of price)
            "min_volatility_atr": 0.3,  # Minimum ATR threshold (% of price)
            "atr_period": 14,  # Period for ATR calculation
            "volatility_lookback": 90,  # Days to look back for volatility normalization
            
            # Risk management parameters
            "risk_per_trade": 0.01,  # 1% account risk per trade
            "max_portfolio_risk": 0.05,  # 5% maximum portfolio risk
            "max_correlated_exposure": 0.03,  # 3% maximum for correlated pairs
            "correlation_threshold": 0.7,  # Threshold above which pairs are considered correlated
            "correlation_lookback": 180,  # Days to calculate correlation
            
            # Position parameters
            "min_position_size": 0.01,  # Minimum position size in lots
            "max_position_size": 2.0,  # Maximum position size in lots per pair
            "max_total_position_size": 10.0,  # Maximum total position size
            "carry_to_risk_min_ratio": 1.5,  # Minimum ratio of annual carry to stop loss
            
            # Exit parameters
            "trailing_stop_atr_multiple": 3.0,  # Trailing stop as multiple of ATR
            "max_drawdown_exit": 0.15,  # 15% drawdown triggers exit review
            
            # Central bank parameters
            "preferred_currencies": ["AUD", "NZD", "CAD", "GBP", "JPY", "EUR", "USD", "CHF"],
            "high_yield_bias": 2.0,  # Bias factor for high-yielding currencies
            "monitor_central_banks": True,  # Whether to monitor central bank events
            
            # Performance tracking
            "target_annual_return": 0.12,  # 12% target annual return
            "max_duration_days": 180,  # Maximum duration to hold a position
            "rebalance_frequency_days": 30,  # How often to rebalance the portfolio
            
            # General parameters
            "timezone": pytz.UTC,
        },
        
        # For aggressive carry trading
        "aggressive": {
            "min_interest_rate_differential": 0.75,
            "use_trend_filter": False,  # Focus purely on carry
            "max_volatility_atr": 2.0,
            "risk_per_trade": 0.015,
            "max_portfolio_risk": 0.08,
            "max_correlated_exposure": 0.05,
            "carry_to_risk_min_ratio": 1.2,
            "trailing_stop_atr_multiple": 4.0,
            "max_drawdown_exit": 0.20,
            "high_yield_bias": 3.0,
        },
        
        # For conservative carry trading
        "conservative": {
            "min_interest_rate_differential": 1.5,
            "use_trend_filter": True,
            "trend_filter_timeframes": ["1d", "1w", "1M"],
            "min_positive_timeframes": 2,
            "max_volatility_atr": 1.0,
            "risk_per_trade": 0.007,
            "max_portfolio_risk": 0.03,
            "max_correlated_exposure": 0.02,
            "correlation_threshold": 0.6,
            "carry_to_risk_min_ratio": 2.0,
            "trailing_stop_atr_multiple": 2.5,
            "max_drawdown_exit": 0.10,
            "high_yield_bias": 1.5,
        }
    }
)
class ForexCarryTradeStrategy(ForexBaseStrategy):
    """
    A strategy that capitalizes on interest rate differentials between currencies.
    
    This strategy buys high-interest currencies and sells low-interest currencies
    to earn the interest rate differential (carry), while applying filters for
    trend, volatility, and correlation to manage risk.
    """
    
    def __init__(self, session=None):
        """
        Initialize the carry trade strategy.
        
        Args:
            session: Trading session object with configuration
        """
        super().__init__(session)
        self.name = "ForexCarryTrade"
        self.description = "Forex Carry Trade Strategy"
        self.logger = logging.getLogger(__name__)
        
        # Interest rate data
        self.interest_rates = {}  # Currency -> interest rate mapping
        self.last_interest_rate_update = None
        
        # Correlation data
        self.correlation_matrix = None
        self.last_correlation_update = None
        
        # Currency pairs and their characteristics
        self.pairs_data = {}  # Pair -> data dictionary
        
        # Portfolio tracking
        self.active_positions = {}  # Symbol -> position data
        self.portfolio_exposure = 0.0  # Total portfolio exposure
        
        # Performance metrics
        self.metrics = {
            "total_carry_earned": 0.0,
            "drawdown": 0.0,
            "max_drawdown": 0.0,
            "carry_to_risk_ratio": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0
        }
        
    def initialize(self) -> None:
        """Initialize the strategy and load required data."""
        super().initialize()
        
        # Initialize interest rate data
        self._update_interest_rates()
        
        # Initialize correlation data
        self._update_correlation_matrix()
        
        # Initialize currency pair data
        self._initialize_pairs_data()
        
        self.logger.info(f"Initialized {self.name} strategy")
    
    def _update_interest_rates(self) -> None:
        """Update the interest rate data for all currencies."""
        # In a real system, this would pull data from an API or database
        # For now, we'll use sample data
        
        # Sample interest rates (% annual) - these would be updated from a data source
        sample_rates = {
            "USD": 5.25,  # Fed funds rate
            "EUR": 3.75,  # ECB rate
            "GBP": 5.00,  # Bank of England rate
            "JPY": 0.10,  # Bank of Japan rate
            "AUD": 4.35,  # RBA rate
            "NZD": 5.50,  # RBNZ rate
            "CAD": 4.50,  # BOC rate
            "CHF": 1.75,  # SNB rate
        }
        
        # Update internal interest rate data
        self.interest_rates = sample_rates
        self.last_interest_rate_update = datetime.now(self.parameters["timezone"])
        
        # Calculate and log the most attractive pairs
        pairs_by_differential = self._calculate_interest_rate_differentials()
        top_pairs = sorted(pairs_by_differential.items(), key=lambda x: x[1], reverse=True)[:5]
        
        self.logger.info(f"Updated interest rates - top 5 pairs by differential:")
        for pair, diff in top_pairs:
            self.logger.info(f"  {pair}: {diff:.2f}%")
    
    def _calculate_interest_rate_differentials(self) -> Dict[str, float]:
        """
        Calculate interest rate differentials for all currency pairs.
        
        Returns:
            Dictionary mapping currency pairs to their interest rate differentials
        """
        differentials = {}
        
        # Get all currency pairs from the list of preferred currencies
        currencies = self.parameters["preferred_currencies"]
        for i, base in enumerate(currencies):
            for quote in currencies[i+1:]:
                # Get interest rates
                base_rate = self.interest_rates.get(base, 0.0)
                quote_rate = self.interest_rates.get(quote, 0.0)
                
                # Calculate differential
                differential = base_rate - quote_rate
                
                # Store in both directions
                differentials[f"{base}{quote}"] = differential
                differentials[f"{quote}{base}"] = -differential
        
        return differentials
    
    def _update_correlation_matrix(self) -> None:
        """
        Update the correlation matrix for currency pairs.
        This is used to manage portfolio risk by avoiding highly correlated positions.
        """
        # In a real implementation, this would calculate actual correlations from price data
        # For now, we'll use a simplified approximation based on currency relationships
        
        # Get all currency pairs from the list of preferred currencies
        currencies = self.parameters["preferred_currencies"]
        pairs = []
        
        for i, base in enumerate(currencies):
            for quote in currencies[i+1:]:
                pairs.append(f"{base}{quote}")
        
        # Initialize an empty correlation matrix
        n_pairs = len(pairs)
        correlation_matrix = pd.DataFrame(np.eye(n_pairs), index=pairs, columns=pairs)
        
        # Fill the correlation matrix with approximate values
        # Pairs that share currencies are more correlated
        for i, pair1 in enumerate(pairs):
            for j, pair2 in enumerate(pairs):
                if i == j:
                    continue  # Skip diagonal (already set to 1.0)
                
                # Extract currencies
                base1, quote1 = pair1[:3], pair1[3:6]  # Simple extraction, assumes 6-char pairs
                base2, quote2 = pair2[:3], pair2[3:6]
                
                # Count shared currencies
                shared = 0
                if base1 == base2 or base1 == quote2:
                    shared += 1
                if quote1 == base2 or quote1 == quote2:
                    shared += 1
                
                # Set correlation based on shared currencies
                if shared == 2:  # Same pair in reverse order (EURUSD vs USDEUR)
                    correlation = -0.95
                elif shared == 1:  # One shared currency
                    correlation = 0.7 if (base1 == base2 or quote1 == quote2) else -0.3
                else:  # No shared currencies
                    correlation = 0.2  # Some base correlation due to global factors
                
                correlation_matrix.loc[pair1, pair2] = correlation
                correlation_matrix.loc[pair2, pair1] = correlation  # Symmetric
        
        self.correlation_matrix = correlation_matrix
        self.last_correlation_update = datetime.now(self.parameters["timezone"])
        
        self.logger.info(f"Updated correlation matrix for {n_pairs} currency pairs")
    
    def _initialize_pairs_data(self) -> None:
        """
        Initialize data for currency pairs, including carry, volatility metrics, and trend status.
        """
        # Get all currency pairs from the list of preferred currencies
        currencies = self.parameters["preferred_currencies"]
        self.pairs_data = {}
        
        differentials = self._calculate_interest_rate_differentials()
        
        for i, base in enumerate(currencies):
            for quote in currencies[i+1:]:
                # Build pairs in both directions
                for pair in [f"{base}{quote}", f"{quote}{base}"]:
                    # Initialize with default values
                    self.pairs_data[pair] = {
                        "base": pair[:3],
                        "quote": pair[3:6],
                        "interest_differential": differentials.get(pair, 0.0),
                        "volatility": 0.0,  # Will be updated with actual data
                        "trend": {
                            "1d": 0,  # 1 = uptrend, 0 = neutral, -1 = downtrend
                            "1w": 0,
                            "1M": 0
                        },
                        "carry_to_risk": 0.0,  # Will be calculated
                        "last_update": datetime.now(self.parameters["timezone"]),
                        "suitable_for_carry": False  # Will be evaluated
                    }
        
        self.logger.info(f"Initialized data for {len(self.pairs_data)} currency pairs")
    
    def _update_pair_data(self, pair: str, data: pd.DataFrame) -> None:
        """
        Update metrics for a specific currency pair.
        
        Args:
            pair: Currency pair symbol
            data: Market data for the pair
        """
        if pair not in self.pairs_data or data.empty:
            return
        
        pair_data = self.pairs_data[pair]
        
        # Update volatility (ATR)
        atr = self._calculate_atr(data, self.parameters["atr_period"])
        current_price = data["close"].iloc[-1]
        atr_percent = (atr / current_price) * 100
        pair_data["volatility"] = atr_percent
        
        # Update trend status
        for timeframe in self.parameters["trend_filter_timeframes"]:
            if timeframe == "1d":
                days = 1
            elif timeframe == "1w":
                days = 7
            elif timeframe == "1M":
                days = 30
            else:
                continue
                
            # Simple trend detection based on current vs past price
            lookback = min(len(data), days * 24)  # Assuming hourly data
            if lookback < 2:
                continue
                
            current_price = data["close"].iloc[-1]
            past_price = data["close"].iloc[-lookback]
            
            if current_price > past_price * 1.01:  # 1% higher
                pair_data["trend"][timeframe] = 1  # Uptrend
            elif current_price < past_price * 0.99:  # 1% lower
                pair_data["trend"][timeframe] = -1  # Downtrend
            else:
                pair_data["trend"][timeframe] = 0  # Neutral
        
        # Calculate carry-to-risk ratio
        interest_differential = pair_data["interest_differential"]
        atr_multiple = self.parameters["trailing_stop_atr_multiple"]
        stop_loss_percent = atr_percent * atr_multiple
        
        # Annual carry vs risk of stop loss
        if stop_loss_percent > 0:
            pair_data["carry_to_risk"] = abs(interest_differential) / stop_loss_percent
        else:
            pair_data["carry_to_risk"] = 0.0
        
        # Determine if pair is suitable for carry trade
        min_diff = self.parameters["min_interest_rate_differential"]
        min_carry_to_risk = self.parameters["carry_to_risk_min_ratio"]
        max_vol = self.parameters["max_volatility_atr"]
        min_vol = self.parameters["min_volatility_atr"]
        
        # Check trend if trend filter is enabled
        trend_suitable = True
        if self.parameters["use_trend_filter"]:
            positive_timeframes = sum(1 for tf, val in pair_data["trend"].items() 
                                    if (val > 0 and interest_differential > 0) or 
                                       (val < 0 and interest_differential < 0))
            trend_suitable = positive_timeframes >= self.parameters["min_positive_timeframes"]
        
        # Final suitability check
        pair_data["suitable_for_carry"] = (
            abs(interest_differential) >= min_diff and
            pair_data["carry_to_risk"] >= min_carry_to_risk and
            min_vol <= atr_percent <= max_vol and
            trend_suitable
        )
        
        pair_data["last_update"] = datetime.now(self.parameters["timezone"])
    
    def generate_signals(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate carry trade signals based on interest rate differentials and market conditions.
        
        Args:
            data_dict: Dictionary of market data for different pairs
            
        Returns:
            Dictionary of trading signals
        """
        signals = {}
        current_time = datetime.now(self.parameters["timezone"])
        
        # Check if we need to update interest rates
        days_since_update = 0
        if self.last_interest_rate_update:
            days_since_update = (current_time - self.last_interest_rate_update).days
            
        if not self.last_interest_rate_update or days_since_update >= self.parameters["interest_rates_refresh_days"]:
            self._update_interest_rates()
        
        # Update pair data for all pairs with available market data
        for pair, data in data_dict.items():
            if pair in self.pairs_data and not data.empty:
                self._update_pair_data(pair, data)
        
        # Get list of suitable pairs for carry trades
        suitable_pairs = [
            pair for pair, data in self.pairs_data.items() 
            if data["suitable_for_carry"] and pair in data_dict
        ]
        
        if not suitable_pairs:
            self.logger.info("No suitable pairs found for carry trades")
            return {}
        
        # Sort by carry-to-risk ratio (best opportunities first)
        suitable_pairs.sort(
            key=lambda p: self.pairs_data[p]["carry_to_risk"] * (
                # Apply high yield bias
                self.parameters["high_yield_bias"] if self.pairs_data[p]["interest_differential"] > 0 else 1.0
            ),
            reverse=True
        )
        
        # Generate signals considering correlation and portfolio constraints
        remaining_risk = self.parameters["max_portfolio_risk"]
        used_pairs = set()  # Track pairs we've already considered
        
        for pair in suitable_pairs:
            # Skip if we've already used this pair
            if pair in used_pairs:
                continue
                
            # Get pair data
            pair_data = self.pairs_data[pair]
            interest_differential = pair_data["interest_differential"]
            
            # Determine trade direction based on interest differential
            # For carry trades, we want to buy the high interest currency and sell the low interest currency
            if interest_differential > 0:
                signal_type = "buy"  # Buy base, sell quote
            else:
                signal_type = "sell"  # Sell base, buy quote
            
            # Check if adding this position would exceed our correlation constraints
            # For each existing position, check correlation with the new pair
            skip_due_to_correlation = False
            correlation_exposure = 0.0
            
            for existing_pair in self.active_positions.keys():
                if existing_pair in self.correlation_matrix.index and pair in self.correlation_matrix.columns:
                    correlation = abs(self.correlation_matrix.loc[existing_pair, pair])
                    if correlation >= self.parameters["correlation_threshold"]:
                        # Calculate contribution to correlated exposure
                        existing_exposure = self.active_positions[existing_pair].get("risk_exposure", 0.0)
                        correlation_exposure += existing_exposure * correlation
                        
                        # If already at the correlated exposure limit, skip this pair
                        if correlation_exposure >= self.parameters["max_correlated_exposure"]:
                            skip_due_to_correlation = True
                            break
            
            if skip_due_to_correlation:
                self.logger.info(f"Skipping {pair} due to correlation constraints")
                continue
            
            # Calculate position size and risk exposure
            risk_per_trade = self.parameters["risk_per_trade"]
            if risk_per_trade > remaining_risk:
                risk_per_trade = remaining_risk
            
            # Skip if not enough risk budget remains
            if risk_per_trade <= 0.001:  # Minimum 0.1% risk
                continue
                
            # Calculate position size based on risk and volatility
            position_size = self._calculate_position_size(pair, data_dict[pair], risk_per_trade)
            
            # Adjust for carry quality (better carry-to-risk gets larger positions)
            carry_quality_factor = min(1.5, pair_data["carry_to_risk"] / self.parameters["carry_to_risk_min_ratio"])
            position_size *= carry_quality_factor
            
            # Apply position size limits
            position_size = max(self.parameters["min_position_size"], 
                              min(self.parameters["max_position_size"], position_size))
            
            # Create signal
            current_price = data_dict[pair]["close"].iloc[-1] if not data_dict[pair].empty else 0
            if current_price <= 0:
                continue
                
            # Calculate stop loss and take profit levels
            atr = self._calculate_atr(data_dict[pair], self.parameters["atr_period"])
            atr_multiple = self.parameters["trailing_stop_atr_multiple"]
            stop_loss_amount = atr * atr_multiple
            
            if signal_type == "buy":
                stop_loss_price = current_price - stop_loss_amount
                take_profit_price = None  # Carry trades typically use trailing stops instead of take profit
            else:  # signal_type == "sell"
                stop_loss_price = current_price + stop_loss_amount
                take_profit_price = None
            
            # Create the signal
            signal = Signal(
                symbol=pair,
                signal_type=signal_type,
                entry_price=current_price,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                size=position_size,
                timestamp=current_time,
                timeframe="1d",  # Carry trades typically use daily or weekly timeframes
                strategy=self.name,
                strength=min(1.0, pair_data["carry_to_risk"] / 5.0),  # Normalize to 0-1 range
                metadata={
                    "interest_differential": interest_differential,
                    "annual_carry_percent": abs(interest_differential),
                    "volatility_atr_percent": pair_data["volatility"],
                    "carry_to_risk_ratio": pair_data["carry_to_risk"],
                    "trade_type": "carry",
                    "base_currency": pair_data["base"],
                    "quote_currency": pair_data["quote"],
                    "max_holding_days": self.parameters["max_duration_days"],
                    "entry_date": current_time.date().isoformat()
                }
            )
            
            # Add to signals and bookkeeping
            signals[pair] = signal
            used_pairs.add(pair)
            remaining_risk -= risk_per_trade
            
            # Add to active positions for tracking
            self.active_positions[pair] = {
                "entry_time": current_time,
                "entry_price": current_price,
                "direction": signal_type,
                "size": position_size,
                "risk_exposure": risk_per_trade,
                "interest_differential": interest_differential,
                "stop_loss": stop_loss_price
            }
            
            # Stop if we've used our risk budget or reached position count limits
            if remaining_risk <= 0.001 or len(signals) >= 5:  # Hard limit of 5 carry positions
                break
            
        if signals:
            self.logger.info(f"Generated {len(signals)} carry trade signals")
            
        return signals
    
    def _calculate_position_size(self, pair: str, data: pd.DataFrame, risk_amount: float) -> float:
        """
        Calculate position size based on risk amount and volatility.
        
        Args:
            pair: Currency pair
            data: Market data
            risk_amount: Amount to risk (as fraction of account)
            
        Returns:
            Position size in lots
        """
        if data.empty:
            return self.parameters["min_position_size"]
        
        # Get ATR for volatility-based stop calculation
        atr = self._calculate_atr(data, self.parameters["atr_period"])
        current_price = data["close"].iloc[-1]
        
        # Calculate stop loss distance based on ATR
        atr_multiple = self.parameters["trailing_stop_atr_multiple"]
        stop_loss_amount = atr * atr_multiple
        
        # Calculate account value at risk
        account_balance = self.session.account_balance
        account_risk_amount = account_balance * risk_amount
        
        # Calculate pip value and position size
        # For simplicity, we'll use a standard pip value calculation
        base_currency = pair[:3]
        quote_currency = pair[3:6]
        account_currency = "USD"  # Default; would be configurable in production
        
        # Standard values for pip calculation
        standard_lot_size = 100000  # 1 standard lot
        point_value = 0.0001 if "JPY" not in pair else 0.01
        pip_value_in_quote = point_value * standard_lot_size
        
        # Convert to account currency if needed (simplified assumption)
        pip_value_in_account = pip_value_in_quote
        if quote_currency != account_currency:
            # In production, would get conversion rate from market data
            conversion_rate = 1.0  # Placeholder
            pip_value_in_account = pip_value_in_quote * conversion_rate
        
        # Calculate stop distance in pips
        stop_pips = stop_loss_amount / point_value
        
        # Calculate position size in standard lots
        if stop_pips > 0 and pip_value_in_account > 0:
            position_size = account_risk_amount / (stop_pips * pip_value_in_account)
        else:
            position_size = 0
        
        # Ensure within limits
        position_size = max(self.parameters["min_position_size"], 
                          min(self.parameters["max_position_size"], position_size))
        
        return position_size
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) for volatility assessment.
        
        Args:
            data: OHLCV data
            period: ATR period
            
        Returns:
            ATR value
        """
        if len(data) < period:
            return 0.0
            
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # Calculate true range components
        high_low = high - low
        high_close = abs(high - close.shift(1))
        low_close = abs(low - close.shift(1))
        
        # Combine for true range
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def _update_performance_metrics(self, position: Dict[str, Any], current_value: float) -> None:
        """
        Update performance metrics for a position.
        
        Args:
            position: Position data
            current_value: Current position value
        """
        # Calculate profit/loss
        entry_value = position["entry_price"]
        direction = position["direction"]
        size = position["size"]
        
        # Calculate pip value (simplified)
        pair = position.get("symbol", "")
        point_value = 0.0001 if "JPY" not in pair else 0.01
        standard_lot = 100000
        pip_value = point_value * standard_lot * size
        
        # Calculate PnL
        if direction == "buy":
            profit_pips = (current_value - entry_value) / point_value
        else:  # direction == "sell"
            profit_pips = (entry_value - current_value) / point_value
            
        profit_amount = profit_pips * pip_value
        
        # Calculate carry earned
        days_held = (datetime.now(self.parameters["timezone"]) - position["entry_time"]).days
        interest_diff = position["interest_differential"]
        annual_carry_pct = abs(interest_diff)
        
        # Daily carry rate (simple approximation)
        daily_carry_pct = annual_carry_pct / 365.0
        
        # Calculate carry amount based on position size and holding time
        # Assuming position value is in standard lots
        position_value = size * standard_lot * entry_value
        carry_earned = position_value * daily_carry_pct * days_held
        
        # Adjust sign based on direction
        if (direction == "buy" and interest_diff < 0) or (direction == "sell" and interest_diff > 0):
            carry_earned = -carry_earned
            
        # Update metrics
        self.metrics["total_carry_earned"] += carry_earned
        
        # Update drawdown
        account_balance = self.session.account_balance
        if profit_amount < 0:
            drawdown_pct = abs(profit_amount) / account_balance
            self.metrics["drawdown"] = max(self.metrics["drawdown"], drawdown_pct)
            self.metrics["max_drawdown"] = max(self.metrics["max_drawdown"], drawdown_pct)
    
    def check_exit_conditions(self, position: Dict[str, Any], data: pd.DataFrame) -> bool:
        """
        Check if a carry trade position should be exited.
        
        Args:
            position: Current position information
            data: Latest market data
            
        Returns:
            True if position should be exited, False otherwise
        """
        if data.empty:
            return False
            
        # Extract position details
        symbol = position["symbol"]
        entry_time_str = position.get("metadata", {}).get("entry_date")
        entry_price = position["entry_price"]
        direction = position["signal_type"]  # 'buy' or 'sell'
        stop_loss = position["stop_loss"]
        
        # Current market data
        current_price = data["close"].iloc[-1]
        current_time = datetime.now(self.parameters["timezone"])
        
        # Check if stop loss triggered
        stop_triggered = False
        if direction == "buy" and current_price <= stop_loss:
            stop_triggered = True
            self.logger.info(f"Exiting {symbol} carry trade due to stop loss triggered")
            return True
        elif direction == "sell" and current_price >= stop_loss:
            stop_triggered = True
            self.logger.info(f"Exiting {symbol} carry trade due to stop loss triggered")
            return True
        
        # Check maximum holding time
        if entry_time_str:
            try:
                entry_date = date.fromisoformat(entry_time_str)
                days_held = (current_time.date() - entry_date).days
                max_duration = position.get("metadata", {}).get("max_holding_days", 
                                                           self.parameters["max_duration_days"])
                
                if days_held >= max_duration:
                    self.logger.info(f"Exiting {symbol} carry trade due to maximum holding period ({days_held} days)")
                    return True
            except (ValueError, TypeError):
                pass
        
        # Check for rebalancing (if it's rebalance day)
        days_since_update = 0
        if self.last_interest_rate_update:
            days_since_update = (current_time - self.last_interest_rate_update).days
            
        if days_since_update >= self.parameters["rebalance_frequency_days"]:
            # Update pair data first
            if symbol in self.pairs_data:
                self._update_pair_data(symbol, data)
                
                # Check if still suitable for carry
                if not self.pairs_data[symbol]["suitable_for_carry"]:
                    self.logger.info(f"Exiting {symbol} carry trade during rebalance - no longer suitable")
                    return True
        
        # Check for interest rate differential reversal
        # This would require fresh interest rate data, which we assume is updated periodically
        if symbol in self.pairs_data:
            current_diff = self.pairs_data[symbol]["interest_differential"]
            original_diff = position.get("metadata", {}).get("interest_differential", 0.0)
            
            # If differential reverses sign, exit the trade
            if current_diff * original_diff < 0:  # Different signs
                self.logger.info(f"Exiting {symbol} carry trade due to interest rate differential reversal")
                return True
        
        # Check for deteriorating carry-to-risk ratio
        if symbol in self.pairs_data:
            current_ratio = self.pairs_data[symbol]["carry_to_risk"]
            min_ratio = self.parameters["carry_to_risk_min_ratio"] * 0.7  # 70% of minimum
            
            if current_ratio < min_ratio:
                self.logger.info(f"Exiting {symbol} carry trade due to poor carry-to-risk ratio ({current_ratio:.2f})")
                return True
        
        # Check for excessive drawdown
        if symbol in self.active_positions:
            position_data = self.active_positions[symbol]
            initial_value = position_data["entry_price"]
            
            # Calculate current drawdown
            if direction == "buy":
                change = (current_price - initial_value) / initial_value
            else:  # direction == "sell"
                change = (initial_value - current_price) / initial_value
                
            if change < -self.parameters["max_drawdown_exit"]:
                self.logger.info(f"Exiting {symbol} carry trade due to excessive drawdown ({change*100:.1f}%)")
                return True
        
        # Check for trailing stop if implemented
        # For simplicity, we'll use a basic implementation
        if symbol in self.active_positions:
            # Update the active position with current data for future reference
            self.active_positions[symbol]["current_price"] = current_price
            
            # Update performance metrics
            self._update_performance_metrics(self.active_positions[symbol], current_price)
        
        return False
    
    def _should_rebalance_portfolio(self) -> bool:
        """
        Determine if the portfolio should be rebalanced based on timing and conditions.
        
        Returns:
            True if rebalance is needed, False otherwise
        """
        # Check if it's time for a scheduled rebalance
        current_time = datetime.now(self.parameters["timezone"])
        days_since_update = 0
        
        if self.last_interest_rate_update:
            days_since_update = (current_time - self.last_interest_rate_update).days
            
        if days_since_update >= self.parameters["rebalance_frequency_days"]:
            return True
        
        # Check if performance metrics indicate need for rebalance
        if self.metrics["drawdown"] > self.parameters["max_drawdown_exit"] * 0.7:
            return True
        
        # Check if interest rate environment has changed
        # This would require comparing current to last rates, which we're not tracking in detail
        
        return False
