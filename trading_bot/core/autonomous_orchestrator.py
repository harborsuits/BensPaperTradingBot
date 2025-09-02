# trading_bot/core/autonomous_orchestrator.py
"""
AutonomousOrchestrator: The robust, smart, and future-proof brain that ties together data, strategies, ML/AI, and pipeline execution.
- Uses CentralDataHub for all data needs
- Uses StrategyRegistry for all strategy logic
- Supports symbol discovery, strategy selection, parameter optimization, backtesting, evaluation, and improvement loop
- Returns results in a structured format for UI or paper trading
"""

import logging
from typing import List, Dict, Any, Optional
from trading_bot.core.data_hub import CentralDataHub
from trading_bot.core.strategy_registry import StrategyRegistry
import datetime

logger = logging.getLogger("AutonomousOrchestrator")

class AutonomousOrchestrator:
    def __init__(self, data_hub: CentralDataHub, benchmark: Optional[Dict[str, float]] = None):
        self.data_hub = data_hub
        self.benchmark = benchmark or {"sharpe_ratio": 1.0, "total_return": 10.0}
        self.results = []

    def discover_symbols(self, top_n=5, asset_types=None) -> List[Dict[str, Any]]:
        """Autonomously discover top candidate symbols across different asset classes.
        
        Args:
            top_n: Number of top candidates to return
            asset_types: List of asset types to include, e.g. ['stock', 'crypto']. If None, all types.
            
        Returns:
            List of dictionaries containing symbol, asset_type, and discovery score
        """
        if asset_types is None:
            asset_types = ['stock', 'crypto', 'forex', 'options']
            
        candidates = []
        
        # Discover stock opportunities
        if 'stock' in asset_types:
            # For demo, use a fixed universe - in production, pull from data_hub
            stock_universe = ["AAPL", "MSFT", "TSLA", "SPY", "GOOGL", "AMZN", "NFLX", "META"]
            for symbol in stock_universe:
                sentiment = self.data_hub.get_sentiment(symbol)
                indicators = self.data_hub.get_indicators(symbol)
                score = sentiment.get("overall", 0) * indicators.get("volatility", 1)
                candidates.append({
                    "symbol": symbol, 
                    "asset_type": "stock",
                    "score": score,
                    "discovery_reason": "High sentiment and volatility"
                })
        
        # Discover crypto opportunities
        if 'crypto' in asset_types:
            crypto_universe = ["BTC/USD", "ETH/USD", "SOL/USD", "DOT/USD", "LINK/USD"]
            for symbol in crypto_universe:
                sentiment = self.data_hub.get_sentiment(symbol)
                indicators = self.data_hub.get_indicators(symbol)
                # For crypto, prioritize momentum and volume
                score = (sentiment.get("overall", 0) * 0.6 + 
                         indicators.get("momentum", 0) * 0.3 + 
                         indicators.get("volume_change", 0) * 0.1)
                candidates.append({
                    "symbol": symbol, 
                    "asset_type": "crypto",
                    "score": score,
                    "discovery_reason": "Strong momentum and increasing volume" 
                })
        
        # Discover forex opportunities
        if 'forex' in asset_types:
            forex_universe = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]
            for symbol in forex_universe:
                # For forex, consider interest rate differentials and economic indicators
                econ_indicators = self.data_hub.get_economic_indicators(symbol.split('/')[0])
                score = econ_indicators.get("interest_rate_differential", 0.01) * 100
                candidates.append({
                    "symbol": symbol, 
                    "asset_type": "forex",
                    "score": score,
                    "discovery_reason": "Favorable interest rate differential" 
                })
        
        # Discover options opportunities
        if 'options' in asset_types:
            # Options require underlying stock symbols with high option volume
            options_universe = ["AAPL", "SPY", "QQQ", "TSLA", "AMZN"]
            for base_symbol in options_universe:
                # In a real implementation, find actual option contracts with good liquidity
                # For demo, just use the base symbol with some naming convention
                implied_vol = self.data_hub.get_indicators(base_symbol).get("implied_volatility", 0.3)
                score = implied_vol * 100  # Higher IV = higher score
                # For demo, create a pseudo options symbol
                option_symbol = f"{base_symbol}_CALL_+5%_30D"  # This would be a real option contract in prod
                candidates.append({
                    "symbol": option_symbol, 
                    "asset_type": "options",
                    "underlying": base_symbol,
                    "score": score,
                    "discovery_reason": "High implied volatility environment" 
                })
        
        # Sort by score and return top N discoveries
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = candidates[:top_n]
        
        logger.info(f"Discovered top symbols: {[c['symbol'] for c in top_candidates]}")
        return top_candidates

    def detect_asset_type(self, symbol: str, context: Dict[str, Any]) -> str:
        """Detect asset type based on symbol and market context.
        
        Determines whether the security is a stock, crypto, forex, or options contract.
        
        Args:
            symbol: The security symbol to identify
            context: Market context including metadata about the security
            
        Returns:
            str: Asset type ('stock', 'crypto', 'forex', 'options', or 'unknown')
        """
        # Check if asset type is already provided in context
        if 'asset_type' in context and context['asset_type']:
            return context['asset_type']
            
        # Otherwise, detect based on symbol patterns and available data
        
        # Crypto detection: typically contains BTC, ETH, or ends with USDT, USD, etc.
        if ('BTC' in symbol or 'ETH' in symbol or symbol.endswith('USDT') or 
            symbol.endswith('USD') and len(symbol) < 8):
            return 'crypto'
            
        # Forex detection: typically currency pairs like EUR/USD, GBP/JPY
        elif '/' in symbol and len(symbol) <= 7:
            return 'forex'
            
        # Options detection: complicated symbols with expiry dates and strikes
        elif len(symbol) > 15 or context.get('option_data') is not None:
            return 'options'
            
        # Default to stock if none of the above
        else:
            return 'stock'
    
    def select_strategy(self, symbol: str, context: Dict[str, Any]) -> str:
        """Select the best-fit strategy for a symbol based on asset type, indicators and sentiment.
        
        This enhanced method first detects the asset type, then selects an appropriate
        strategy from the asset-specific strategies available in the registry.
        
        Args:
            symbol: Trading symbol to select strategy for
            context: Market context including indicators, sentiment, and price data
            
        Returns:
            str: Selected strategy name from the registry
        """
        from trading_bot.core.strategy_registry import StrategyRegistry
        
        # Step 1: Detect asset type
        asset_type = self.detect_asset_type(symbol, context)
        context['asset_type'] = asset_type  # Store for future reference
        logger.info(f"Detected asset type for {symbol}: {asset_type}")
        
        # Step 2: Apply asset-specific strategy selection logic
        if asset_type == 'stock':
            return self._select_stock_strategy(symbol, context)
        elif asset_type == 'crypto':
            return self._select_crypto_strategy(symbol, context)
        elif asset_type == 'forex':
            return self._select_forex_strategy(symbol, context)
        elif asset_type == 'options':
            return self._select_options_strategy(symbol, context)
        else:
            # Fallback to generic strategies
            logger.warning(f"Unknown asset type for {symbol}, using generic strategy selection")
            if context["sentiment"].get("overall", 0) > 0.5 and context["indicators"].get("momentum", 0) > 0.5:
                return "momentum"
            elif context["indicators"].get("volatility", 0) > 2:
                return "mean_reversion"
            else:
                return "trend_following"
    
    def _select_stock_strategy(self, symbol: str, context: Dict[str, Any]) -> str:
        """Select appropriate stock-specific strategy based on market context."""
        # Get market indicators
        indicators = context.get("indicators", {})
        sentiment = context.get("sentiment", {})
        price_data = context.get("price_data", None)
        
        # Strategy selection logic for stocks
        if indicators.get("volatility", 0) > 2.5:
            return "stock_bollinger"  # High volatility -> Bollinger band strategy
        elif indicators.get("rsi", 50) < 30 or indicators.get("rsi", 50) > 70:
            return "stock_rsi_reversion"  # Extreme RSI -> Mean reversion
        elif indicators.get("momentum", 0) > 0.7:
            return "stock_momentum"  # Strong momentum
        elif indicators.get("volume_spike", False):
            return "stock_volume_breakout"  # Volume spike -> Breakout
        else:
            return "stock_ma_cross"  # Default to MA cross strategy
    
    def _select_crypto_strategy(self, symbol: str, context: Dict[str, Any]) -> str:
        """Select appropriate crypto-specific strategy based on market context."""
        # Get market indicators
        indicators = context.get("indicators", {})
        sentiment = context.get("sentiment", {})
        
        # Strategy selection logic for crypto
        if 'onchain_metrics' in context and context['onchain_metrics'].get('whale_activity', 0) > 0.7:
            return "crypto_token_flow"  # Whale activity -> Token flow analysis
        elif indicators.get("volatility", 0) > 3.0:
            return "crypto_range"  # High volatility -> Range trading
        elif indicators.get("momentum", 0) > 0.6:
            return "crypto_momentum"  # Strong momentum
        else:
            return "crypto_rsi"  # Default to RSI-based strategy
    
    def _select_forex_strategy(self, symbol: str, context: Dict[str, Any]) -> str:
        """Select appropriate forex-specific strategy based on market context."""
        # Get market indicators
        indicators = context.get("indicators", {})
        sentiment = context.get("sentiment", {})
        interest_rates = context.get("interest_rates", {})
        
        # Strategy selection logic for forex
        if interest_rates.get("differential", 0) > 2.0:
            return "forex_interest_rate"  # High rate differential -> Carry trade
        elif indicators.get("range_bound", False):
            return "forex_range"  # Range-bound market -> Range strategy
        elif indicators.get("trends", 0) > 0.6:
            return "forex_trend"  # Strong trend
        else:
            return "forex_support_resistance"  # Default to support/resistance
    
    def _select_options_strategy(self, symbol: str, context: Dict[str, Any]) -> str:
        """Select appropriate options-specific strategy based on market context."""
        # Get market indicators
        indicators = context.get("indicators", {})
        sentiment = context.get("sentiment", {})
        volatility = context.get("implied_volatility", 0.3)
        
        # Strategy selection logic for options
        if volatility > 0.4:  # High implied volatility
            return "options_cash_secured_put"  # Collect premium in high IV
        elif 0.25 <= volatility <= 0.4:  # Medium implied volatility
            return "options_iron_condor"  # Range-bound strategy
        elif indicators.get("bullish", False):
            return "options_bull_call_spread"  # Bullish strategy
        elif indicators.get("bearish", False):
            return "options_bear_put_spread"  # Bearish strategy
        else:
            return "options_wheel"  # Default to wheel strategy

    def optimize_parameters(self, strategy_name: str, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize parameters for a given strategy and symbol using ParameterOptimizer."""
        try:
            import os
            from trading_bot.simulation.parameter_optimizer import ParameterOptimizer, ParameterSpace, OptimizationConfig
            from trading_bot.simulation.trading_simulator import SimulationConfig, SimulationMode, MarketScenario
            from trading_bot.data.data_providers import MockDataProvider

            # 1. Build parameter spaces for each supported strategy
            default_spaces = {
                "momentum": [
                    ParameterSpace(name="fast_period", values=[3, 5, 10], description="Fast MA period"),
                    ParameterSpace(name="slow_period", values=[15, 20, 30], description="Slow MA period"),
                ],
                "trend_following": [
                    ParameterSpace(name="short_ma_period", values=[5, 10, 20], description="Short MA period"),
                    ParameterSpace(name="long_ma_period", values=[20, 30, 50], description="Long MA period"),
                ],
                "mean_reversion": [
                    ParameterSpace(name="period", values=[10, 20, 30], description="Lookback period"),
                    ParameterSpace(name="std_dev_factor", values=[1.5, 2.0, 2.5], description="Std Dev Factor"),
                ],
            }
            parameter_spaces = default_spaces.get(strategy_name, [])
            if not parameter_spaces:
                logger.warning(f"No parameter space defined for strategy '{strategy_name}', using default.")
                parameter_spaces = [ParameterSpace(name="window", values=[10, 20, 30], description="Generic window")]

            # 2. Build OptimizationConfig
            optimization_config = OptimizationConfig(
                parameter_spaces=parameter_spaces,
                target_metric="sharpe_ratio",
                higher_is_better=True,
                max_parallel_jobs=2,
                output_dir=os.path.join("optimization_results", strategy_name, symbol),
                random_seed=42
            )

            # 3. Build SimulationConfig (use context and defaults)
            from datetime import datetime, timedelta
            price_data = context.get("price_data")
            if price_data is not None and hasattr(price_data, 'index'):
                start_date = price_data.index[0] if len(price_data.index) > 0 else datetime.now() - timedelta(days=365)
                end_date = price_data.index[-1] if len(price_data.index) > 0 else datetime.now()
            else:
                start_date = datetime.now() - timedelta(days=365)
                end_date = datetime.now()

            sim_config = SimulationConfig(
                mode=SimulationMode.BACKTEST,
                start_date=start_date,
                end_date=end_date,
                initial_capital=100000.0,
                symbols=[symbol],
                market_scenario=MarketScenario.NORMAL,
                data_frequency="1d",
                slippage_model="fixed",
                slippage_value=0.0,
                commission_model="fixed",
                commission_value=0.0,
                enable_fractional_shares=True,
                random_seed=42
            )

            # 4. Data provider (use a mock or real one as appropriate)
            data_provider = MockDataProvider()

            # 5. Strategy factory
            strategy_cls = StrategyRegistry.get(strategy_name)
            def strategy_factory(symbol, data_provider, risk_manager=None, **params):
                # Accepts params as flat dict
                config = {k: v for k, v in params.items()}
                return strategy_cls(symbol, config=config)

            # 6. Instantiate and run optimizer
            optimizer = ParameterOptimizer(
                base_simulation_config=sim_config,
                optimization_config=optimization_config,
                data_provider=data_provider,
                strategy_factory=strategy_factory,
            )
            logger.info(f"Starting parameter optimization for {strategy_name} on {symbol}")
            result = optimizer.run_grid_search()
            logger.info(f"Optimization complete. Best params: {result.best_parameters}, Score: {result.best_score}")
            return result.best_parameters or {}
        except Exception as e:
            logger.error(f"Parameter optimization failed for {strategy_name} on {symbol}: {e}")
            return {}

    def run_pipeline(self, top_n=5, optimize_params=False, asset_types=None) -> List[Dict[str, Any]]:
        """Run the full autonomous pipeline across multiple asset classes and return structured results.
        
        Args:
            top_n: Number of top candidates to return
            optimize_params: Whether to optimize strategy parameters
            asset_types: List of asset types to include, e.g. ['stock', 'crypto']. If None, all types.
            
        Returns:
            List of dictionaries containing opportunity details and backtesting results
        """
        results = []
        
        # Discover opportunities across asset classes
        discovered_opportunities = self.discover_symbols(top_n=top_n, asset_types=asset_types)
        
        for opportunity in discovered_opportunities:
            symbol = opportunity["symbol"]
            asset_type = opportunity["asset_type"]
            discovery_reason = opportunity.get("discovery_reason", "")
            
            # Build context based on asset type
            context = {
                "asset_type": asset_type,
                "sentiment": self.data_hub.get_sentiment(symbol),
                "indicators": self.data_hub.get_indicators(symbol),
                "price_data": self.data_hub.get_price_data(symbol)
            }
            
            # Add asset-specific context data
            if asset_type == "options":
                underlying = opportunity.get("underlying")
                if underlying:
                    context["underlying_data"] = self.data_hub.get_price_data(underlying)
                    context["implied_volatility"] = self.data_hub.get_indicators(underlying).get("implied_volatility", 0.3)
            elif asset_type == "forex":
                # Add forex-specific context like interest rates
                currencies = symbol.split('/')
                if len(currencies) == 2:
                    context["interest_rates"] = {
                        currencies[0]: self.data_hub.get_economic_indicators(currencies[0]).get("interest_rate", 0),
                        currencies[1]: self.data_hub.get_economic_indicators(currencies[1]).get("interest_rate", 0)
                    }
            elif asset_type == "crypto":
                # Try to get on-chain metrics for crypto
                context["onchain_metrics"] = self.data_hub.get_onchain_metrics(symbol) if hasattr(self.data_hub, "get_onchain_metrics") else {}
            
            # Select the appropriate strategy for this asset type
            strategy_name = self.select_strategy(symbol, context)
            
            # Optimize parameters if requested
            if optimize_params:
                params = self.optimize_parameters(strategy_name, symbol, context)
            else:
                params = {}
                
            # Create and run the strategy
            try:
                strategy_cls = StrategyRegistry.get(strategy_name)
                strategy = strategy_cls(name=symbol, config=params)
                
                # Run backtest with the appropriate data
                backtest_result = strategy.backtest(context["price_data"])
                
                # Evaluate against benchmark
                meets_benchmark = (backtest_result.get("sharpe_ratio", 0) >= self.benchmark["sharpe_ratio"] and
                                  backtest_result.get("total_return", 0) >= self.benchmark["total_return"])
                                  
                # Create entry/exit points based on the strategy
                current_price = context["price_data"].iloc[-1]["close"] if isinstance(context["price_data"], pd.DataFrame) else 100.0
                
                # Use strategy to get stop loss and target - this would call the strategy's methods in real implementation
                stop_loss_pct = -0.05  # Default 5% stop loss
                target_pct = 0.10      # Default 10% target
                
                if hasattr(strategy, "calculate_stop_loss"):
                    stop_loss_pct = strategy.calculate_stop_loss(context["price_data"])
                    
                if hasattr(strategy, "calculate_profit_target"):
                    target_pct = strategy.calculate_profit_target(context["price_data"])
                
                # Create rich result object
                result = {
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "strategy": strategy_name,
                    "discovery_reason": discovery_reason,
                    "params": params,
                    "current_price": current_price,
                    "entry_price": current_price,  # Simplified - would be calculated by strategy
                    "stop_loss": current_price * (1 + stop_loss_pct),
                    "profit_target": current_price * (1 + target_pct),
                    "position_size": self._calculate_position_size(symbol, asset_type, current_price),
                    "metrics": backtest_result,
                    "meets_benchmark": meets_benchmark,
                    "confidence": self._calculate_confidence(backtest_result),
                    "expected_return": backtest_result.get("expected_return", target_pct),
                    "sharpe": backtest_result.get("sharpe_ratio", 1.0),
                    "run_time": datetime.datetime.now().isoformat(),
                    "submission_time": datetime.datetime.now().isoformat(),
                    "id": f"{symbol}_{strategy_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "status": "Ready" if meets_benchmark else "Below Threshold",
                    "evidence": f"Based on {discovery_reason.lower()} with {backtest_result.get('win_rate', 0.5):.1%} win rate"
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Pipeline failed for {symbol} ({asset_type}) with {strategy_name}: {e}")
                # Add failed result with error info
                results.append({
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "strategy": strategy_name,
                    "error": str(e),
                    "run_time": datetime.datetime.now().isoformat(),
                    "status": "Failed"
                })
        
        # Sort results by confidence and expected return
        if results:
            results.sort(key=lambda x: (x.get("confidence", 0) * 0.7 + x.get("expected_return", 0) * 0.3), reverse=True)
            
        self.results = results
        return results
        
    def _calculate_position_size(self, symbol: str, asset_type: str, current_price: float) -> float:
        """Calculate appropriate position size based on asset type and volatility."""
        # This would be enhanced with proper risk management in production
        base_capital = 100000.0  # $100k trading capital
        
        if asset_type == "stock":
            # 1-5% of capital per trade depending on volatility
            return base_capital * 0.03  # 3% allocation
        elif asset_type == "crypto":
            # Lower allocation for crypto due to higher volatility
            return base_capital * 0.02  # 2% allocation
        elif asset_type == "forex":
            # Forex often uses leverage
            return base_capital * 0.05  # 5% allocation 
        elif asset_type == "options":
            # Options typically use smaller allocations
            return base_capital * 0.01  # 1% allocation
        else:
            return base_capital * 0.03  # Default allocation
    
    def _calculate_confidence(self, backtest_result: Dict[str, Any]) -> float:
        """Calculate confidence score based on backtest metrics."""
        # Simplified calculation - would be more sophisticated in production
        sharpe = backtest_result.get("sharpe_ratio", 0)
        win_rate = backtest_result.get("win_rate", 0)
        consistency = backtest_result.get("consistency", 0)
        
        # Weight the factors
        confidence = (
            sharpe * 0.4 +  # 40% weight to Sharpe ratio
            win_rate * 0.4 +  # 40% weight to win rate
            consistency * 0.2  # 20% weight to consistency
        )
        
        # Normalize to 0-1 range
        return min(max(confidence, 0), 1)
