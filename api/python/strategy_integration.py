import logging
import json
import os
import yaml
import numpy as np
from datetime import datetime
from strategy_logic.option_spreads_strategy import OptionSpreadsStrategy
from context_layer.strategy_selector import StrategySelector

logger = logging.getLogger(__name__)

class StrategyIntegration:
    """
    Strategy Integration Layer that defines relationships between equity strategies and 
    options strategies to create an interconnected trading framework.
    """
    
    def __init__(self, config, context_engine=None):
        """Initialize the strategy integration layer."""
        self.config = config
        self.context_engine = context_engine
        self.options_strategy = OptionSpreadsStrategy()
        self.strategy_selector = StrategySelector(config)
        
        # Load integration mapping
        self.integration_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "integration_mapping.json")
        if os.path.exists(self.integration_file):
            with open(self.integration_file, "r") as f:
                self.integration_mapping = json.load(f)
        else:
            self.integration_mapping = self._create_default_mapping()
            # Save the default mapping
            with open(self.integration_file, "w") as f:
                json.dump(self.integration_mapping, f, indent=2)
        
        logger.info(f"Strategy Integration Layer initialized with {len(self.integration_mapping.get('core_to_options', {}))} strategy mappings")
    
    def _create_default_mapping(self):
        """Create default strategy integration mappings based on specified framework."""
        return {
            "core_to_options": {
                "day_trading": {
                    "gap_fill_daytrade": [
                        {"options_strategy": "weekly_options_momentum", "integration_method": "shadow_strategy", "market_condition": "any"},
                        {"options_strategy": "earnings_strangle", "integration_method": "hedge_strategy", "market_condition": "high_volatility"}
                    ],
                    "vwap_bounce": [
                        {"options_strategy": "rsi_ema_reversal", "integration_method": "confirmation_system", "market_condition": "trending"},
                        {"options_strategy": "weekly_options_momentum", "integration_method": "leverage_tool", "market_condition": "trending"}
                    ],
                    "opening_range_breakout": [
                        {"options_strategy": "weekly_options_momentum", "integration_method": "alternative_execution", "market_condition": "volatile"},
                        {"options_strategy": "long_call_vertical", "integration_method": "risk_defined_alternative", "market_condition": "any"}
                    ]
                },
                "swing_trading": {
                    "breakout_swing": [
                        {"options_strategy": "bull_call_spread", "integration_method": "leveraged_alternative", "market_condition": "bullish"},
                        {"options_strategy": "bull_put_spread", "integration_method": "income_generator", "market_condition": "bullish"}
                    ],
                    "pullback_to_moving_average": [
                        {"options_strategy": "bull_put_spread", "integration_method": "primary_strategy", "market_condition": "bullish"},
                        {"options_strategy": "calendar_spread", "integration_method": "enhanced_income", "market_condition": "sideways_to_bullish"}
                    ],
                    "oversold_bounce": [
                        {"options_strategy": "long_call", "integration_method": "maximum_leverage", "market_condition": "bottoming"},
                        {"options_strategy": "bull_call_spread", "integration_method": "defined_risk", "market_condition": "bottoming"}
                    ]
                },
                "options_strategies": {
                    "theta_spread": [
                        {"options_strategy": "iron_condor", "integration_method": "width_adjustment", "market_condition": "sideways"},
                        {"options_strategy": "calendar_spread", "integration_method": "time_exploitation", "market_condition": "sideways"}
                    ],
                    "volatility_squeeze": [
                        {"options_strategy": "long_straddle", "integration_method": "core_execution", "market_condition": "transitioning"},
                        {"options_strategy": "diagonal_spread", "integration_method": "cost_reduction", "market_condition": "low_volatility"}
                    ],
                    "iron_condor": [
                        {"options_strategy": "butterfly_spread", "integration_method": "adjustment_mechanism", "market_condition": "sideways"},
                        {"options_strategy": "jade_lizard", "integration_method": "directional_tilt", "market_condition": "bullish"}
                    ]
                }
            },
            "integration_frameworks": {
                "confirmation_framework": {
                    "intraday_confirmation": {
                        "primary": "gap_fill_daytrade",
                        "confirmation": "rsi_ema_reversal",
                        "execution": [
                            "confirm primary setup first",
                            "require minimum 2 confirmation indicators from secondary strategy",
                            "take position size: 125% of standard if both aligned",
                            "prefer options strategy for execution when confirmed"
                        ]
                    },
                    "swing_confirmation": {
                        "primary": "breakout_swing",
                        "confirmation": "options_chain_analysis",
                        "execution": [
                            "confirm price action in equity first",
                            "verify option chain shows supporting flow (increasing call volume/OI)",
                            "use bull call spreads when confirmed",
                            "sizing: standard for partial confirmation, 125% for full confirmation"
                        ]
                    }
                },
                "conversion_pathways": {
                    "equity_to_options": {
                        "trigger_conditions": [
                            "profit target 50% reached in equity position",
                            "increasing implied volatility",
                            "technical resistance approaching"
                        ],
                        "conversion_rules": [
                            "replace equity with vertical spreads to lock in partial gains",
                            "roll portion (50%) to further OTM options for continued upside",
                            "maintain delta exposure but reduce capital commitment"
                        ]
                    },
                    "options_to_equity": {
                        "trigger_conditions": [
                            "implied volatility contraction",
                            "options approaching 21 DTE",
                            "technical consolidation pattern forming"
                        ],
                        "conversion_rules": [
                            "close in-the-money options positions",
                            "establish equity position at 75% of original delta exposure",
                            "implement covered call strategy if applicable"
                        ]
                    }
                },
                "volatility_integration": {
                    "high_iv_environment": {
                        "emphasis": "premium_selling",
                        "primary_strategies": [
                            "theta_spread",
                            "iron_condor",
                            "bull_put_spread"
                        ],
                        "equity_adjustment": [
                            "reduce equity position sizing by 30%",
                            "prefer mean-reversion setups",
                            "implement covered call/collar strategies on existing positions"
                        ]
                    },
                    "low_iv_environment": {
                        "emphasis": "long_options",
                        "primary_strategies": [
                            "long_call",
                            "bull_call_spread",
                            "volatility_squeeze"
                        ],
                        "equity_adjustment": [
                            "standard equity position sizing",
                            "favor momentum and breakout strategies",
                            "use options as leverage for highest conviction setups"
                        ]
                    },
                    "neutral_iv_environment": {
                        "emphasis": "balanced_approach",
                        "strategy_blend": [
                            "50% standard equity strategies",
                            "30% premium collection strategies",
                            "20% long options strategies"
                        ]
                    }
                }
            },
            "capital_allocation": {
                "bullish_market": {
                    "equity_allocation": 0.6,
                    "options_allocation": {
                        "directional_calls": 0.2,
                        "bull_put_spreads": 0.15,
                        "other_strategies": 0.05
                    },
                    "strategy_weighting": {
                        "highest_weighting": ["breakout_swing", "bull_call_spread"],
                        "secondary_weighting": ["pullback_to_moving_average", "bull_put_spread"]
                    }
                },
                "bearish_market": {
                    "equity_allocation": 0.3,
                    "options_allocation": {
                        "directional_puts": 0.15,
                        "bear_call_spreads": 0.15,
                        "hedging_strategies": 0.1,
                        "cash_reserve": 0.3
                    },
                    "strategy_weighting": {
                        "highest_weighting": ["gap_fill_daytrade", "bear_put_spread"],
                        "secondary_weighting": ["volatility_squeeze", "protective_puts"]
                    }
                },
                "sideways_market": {
                    "equity_allocation": 0.4,
                    "options_allocation": {
                        "premium_selling": 0.35,
                        "volatility_strategies": 0.1,
                        "cash_reserve": 0.15
                    },
                    "strategy_weighting": {
                        "highest_weighting": ["vwap_bounce", "iron_condor"],
                        "secondary_weighting": ["theta_spread", "calendar_spread"]
                    }
                }
            },
            "unified_risk_framework": {
                "position_correlation": {
                    "max_strategy_correlation": 0.7,
                    "calculation": "equity beta + options delta",
                    "sector_limits": "25% maximum exposure to any sector"
                },
                "drawdown_management": {
                    "equity_drawdown_response": [
                        {"threshold": 0.03, "action": "reduce position size by 25%"},
                        {"threshold": 0.05, "action": "pause new equity entries"}
                    ],
                    "options_drawdown_response": [
                        {"threshold": 0.02, "action": "reduce position size by 35%"},
                        {"threshold": 0.04, "action": "close speculative options positions"}
                    ],
                    "combined_response": [
                        "reduce highest delta positions first",
                        "maintain hedged positions longer",
                        "convert directional options to spreads"
                    ]
                },
                "volatility_adjustment": [
                    {"vix_range": [0, 15], "action": "standard position sizing"},
                    {"vix_range": [15, 25], "action": "reduce size by 10%, favor premium selling"},
                    {"vix_range": [25, 100], "action": "reduce size by 25%, favor defined-risk strategies"}
                ]
            }
        }
    
    def get_integration_mapping(self, strategy_name=None, category=None):
        """
        Get integration mapping for a specific strategy or category.
        
        Args:
            strategy_name (str, optional): The core strategy name
            category (str, optional): The strategy category (day_trading, swing_trading, options_strategies)
            
        Returns:
            dict: The integration mapping
        """
        if not strategy_name and not category:
            return self.integration_mapping.get("core_to_options", {})
        
        if category:
            return self.integration_mapping.get("core_to_options", {}).get(category, {})
        
        # Search for strategy in all categories
        for category, strategies in self.integration_mapping.get("core_to_options", {}).items():
            if strategy_name in strategies:
                return strategies[strategy_name]
        
        return []
    
    def get_paired_strategies(self, market_condition, core_strategy=None, integration_method=None):
        """
        Get paired strategies based on market condition and optional core strategy.
        
        Args:
            market_condition (str): Current market condition
            core_strategy (str, optional): The core strategy name
            integration_method (str, optional): Type of integration method to filter by
            
        Returns:
            list: List of paired strategies
        """
        paired_strategies = []
        
        # Check all strategy mappings
        for category, strategies in self.integration_mapping.get("core_to_options", {}).items():
            for strategy_name, pairings in strategies.items():
                # Skip if not matching core strategy when specified
                if core_strategy and strategy_name != core_strategy:
                    continue
                    
                # Check each pairing for market condition match
                for pairing in pairings:
                    pairing_market = pairing.get("market_condition", "")
                    pairing_method = pairing.get("integration_method", "")
                    
                    # Check if market conditions match
                    market_match = (pairing_market == "any" or 
                                   pairing_market == market_condition or
                                   market_condition in pairing_market)
                    
                    # Check if integration method matches if specified
                    method_match = not integration_method or integration_method == pairing_method
                    
                    if market_match and method_match:
                        paired_strategies.append({
                            "core_strategy": strategy_name,
                            "category": category,
                            "options_strategy": pairing.get("options_strategy", ""),
                            "integration_method": pairing_method
                        })
        
        return paired_strategies
    
    def get_integration_framework(self, framework_type, framework_name=None):
        """
        Get integration framework configuration.
        
        Args:
            framework_type (str): Type of framework (confirmation_framework, conversion_pathways, volatility_integration)
            framework_name (str, optional): Specific framework name
            
        Returns:
            dict: Framework configuration
        """
        frameworks = self.integration_mapping.get("integration_frameworks", {}).get(framework_type, {})
        
        if framework_name:
            return frameworks.get(framework_name, {})
        
        return frameworks
    
    def get_capital_allocation(self, market_condition):
        """
        Get capital allocation guidance based on market condition.
        
        Args:
            market_condition (str): Current market condition
            
        Returns:
            dict: Capital allocation framework
        """
        condition_map = {
            "bullish": "bullish_market",
            "bearish": "bearish_market",
            "sideways": "sideways_market",
            "volatile": "sideways_market",  # Default to sideways for volatile
            "high_volatility": "sideways_market"
        }
        
        mapped_condition = condition_map.get(market_condition, "sideways_market")
        return self.integration_mapping.get("capital_allocation", {}).get(mapped_condition, {})
    
    def get_volatility_integration(self, iv_rank):
        """
        Get volatility-based integration framework based on IV rank.
        
        Args:
            iv_rank (float): Current IV rank (0-100)
            
        Returns:
            dict: Volatility integration framework
        """
        frameworks = self.integration_mapping.get("integration_frameworks", {}).get("volatility_integration", {})
        
        if iv_rank > 60:
            return frameworks.get("high_iv_environment", {})
        elif iv_rank < 30:
            return frameworks.get("low_iv_environment", {})
        else:
            return frameworks.get("neutral_iv_environment", {})
    
    def calculate_position_size(self, core_strategy, options_strategy, integration_method, account_value, risk_per_trade=0.01):
        """
        Calculate position size based on integration framework and risk parameters.
        
        Args:
            core_strategy (str): Core strategy name
            options_strategy (str): Options strategy name
            integration_method (str): Integration method
            account_value (float): Account value
            risk_per_trade (float, optional): Base risk per trade as percentage of account
            
        Returns:
            dict: Position sizing recommendations
        """
        # Base position size
        base_position = account_value * risk_per_trade
        
        # Size multipliers based on integration method
        integration_multipliers = {
            "confirmation_system": 1.25,
            "leverage_tool": 0.75,
            "shadow_strategy": 1.0,
            "hedge_strategy": 0.5,
            "alternative_execution": 0.75,
            "risk_defined_alternative": 1.0,
            "leveraged_alternative": 0.75,
            "income_generator": 1.0,
            "primary_strategy": 1.0,
            "enhanced_income": 0.8,
            "maximum_leverage": 0.5,
            "defined_risk": 1.0,
            "width_adjustment": 0.9,
            "time_exploitation": 0.8,
            "core_execution": 1.0,
            "cost_reduction": 0.9,
            "adjustment_mechanism": 0.8,
            "directional_tilt": 0.9
        }
        
        # Get market condition and IV
        market_condition = "sideways"
        iv_rank = 50
        vix_level = 15
        
        if self.context_engine:
            market_indicators = self.context_engine.get_market_indicators()
            market_condition = self.context_engine.strategy_selector.get_market_condition(market_indicators)
            iv_rank = market_indicators.get("iv_rank", 50)
            vix_level = market_indicators.get("vix", 15)
        
        # Get volatility adjustment
        volatility_adjustments = self.integration_mapping.get("unified_risk_framework", {}).get("volatility_adjustment", [])
        vol_multiplier = 1.0
        
        for adjustment in volatility_adjustments:
            vix_range = adjustment.get("vix_range", [0, 100])
            if vix_range[0] <= vix_level < vix_range[1]:
                action = adjustment.get("action", "")
                if "reduce size by" in action:
                    try:
                        reduction = float(action.split("reduce size by")[1].split("%")[0].strip()) / 100
                        vol_multiplier = 1.0 - reduction
                    except:
                        vol_multiplier = 1.0
                break
        
        # Integration method multiplier
        method_multiplier = integration_multipliers.get(integration_method, 1.0)
        
        # Calculate sizes
        equity_size = base_position * method_multiplier * vol_multiplier
        options_size = base_position * method_multiplier * vol_multiplier
        
        # Adjust based on IV environment for options
        if iv_rank > 60:
            # High IV - reduce directional options size, increase premium selling size
            if "buy" in options_strategy or "long" in options_strategy:
                options_size *= 0.7
            elif "sell" in options_strategy or "credit" in options_strategy:
                options_size *= 1.1
        elif iv_rank < 30:
            # Low IV - increase directional options size, decrease premium selling size
            if "buy" in options_strategy or "long" in options_strategy:
                options_size *= 1.1
            elif "sell" in options_strategy or "credit" in options_strategy:
                options_size *= 0.8
        
        # Capital allocation based on market condition
        allocation = self.get_capital_allocation(market_condition)
        
        return {
            "equity_position_size": round(equity_size, 2),
            "options_position_size": round(options_size, 2),
            "account_value": account_value,
            "market_condition": market_condition,
            "iv_rank": iv_rank,
            "vix_level": vix_level,
            "volatility_multiplier": vol_multiplier,
            "integration_multiplier": method_multiplier,
            "recommended_allocation": allocation
        }
    
    def recommend_strategy_integration(self, ticker=None, account_value=None):
        """
        Recommend integrated strategy approach based on current market conditions.
        
        Args:
            ticker (str, optional): Symbol of interest
            account_value (float, optional): Account value for position sizing
            
        Returns:
            dict: Recommended integrated strategy approach
        """
        # Get market indicators and condition
        market_indicators = {}
        market_condition = "sideways"
        iv_rank = 50
        
        if self.context_engine:
            market_indicators = self.context_engine.get_market_indicators()
            market_condition = self.strategy_selector.get_market_condition(market_indicators)
            iv_rank = market_indicators.get("iv_rank", 50)
        
        # Get volatility-based integration
        vol_integration = self.get_volatility_integration(iv_rank)
        
        # Get recommended equity strategies
        equity_strategies = self.strategy_selector.get_recommended_strategies(market_condition, num_strategies=2)
        
        # Get paired strategies for each equity strategy
        integrated_strategies = []
        for strategy in equity_strategies:
            paired = self.get_paired_strategies(market_condition, core_strategy=strategy)
            if paired:
                for pair in paired:
                    integrated_strategies.append({
                        "core_strategy": strategy,
                        "options_strategy": pair.get("options_strategy"),
                        "integration_method": pair.get("integration_method"),
                        "category": pair.get("category")
                    })
        
        # If we don't have integrated strategies, get general recommendations
        if not integrated_strategies:
            integrated_strategies = self.get_paired_strategies(market_condition)
        
        # Sort by category - prioritize day trading in trending/volatile markets, swing in others
        if market_condition in ["trending", "volatile"]:
            integrated_strategies.sort(key=lambda x: 0 if x.get("category") == "day_trading" else 1)
        else:
            integrated_strategies.sort(key=lambda x: 0 if x.get("category") == "swing_trading" else 1)
        
        # Calculate position sizes if account value provided
        if account_value and integrated_strategies:
            top_strategy = integrated_strategies[0]
            sizing = self.calculate_position_size(
                top_strategy.get("core_strategy"),
                top_strategy.get("options_strategy"),
                top_strategy.get("integration_method"),
                account_value
            )
        else:
            sizing = None
        
        # Get strategy stacks if suitable
        strategy_stacks = []
        if iv_rank > 60 and market_condition == "sideways":
            # Good conditions for premium selling stacks
            stack_framework = self.integration_mapping.get("advanced_techniques", {}).get("strategy_stacking", {})
            strategy_stacks = [stack_framework.get("diagonal_calendar_stack", {})]
        
        return {
            "market_condition": market_condition,
            "iv_rank": iv_rank,
            "volatility_integration": vol_integration,
            "integrated_strategies": integrated_strategies[:3],  # Top 3 recommendations
            "position_sizing": sizing,
            "strategy_stacks": strategy_stacks,
            "capital_allocation": self.get_capital_allocation(market_condition)
        }
    
    def should_convert_strategy(self, current_position, market_indicators):
        """
        Check if a strategy conversion is warranted based on current conditions.
        
        Args:
            current_position (dict): Current position details
            market_indicators (dict): Current market indicators
            
        Returns:
            tuple: (should_convert, reason, conversion_type)
        """
        # Get conversion pathways
        pathways = self.get_integration_framework("conversion_pathways")
        
        position_type = current_position.get("type", "")
        profit_pct = current_position.get("profit_pct", 0)
        days_to_expiration = current_position.get("days_to_expiration", 30)
        iv_change = current_position.get("iv_change", 0)
        
        # Check equity to options conversion
        if position_type == "equity":
            equity_to_options = pathways.get("equity_to_options", {})
            trigger_conditions = equity_to_options.get("trigger_conditions", [])
            
            # Check profit target
            profit_target_met = profit_pct >= 0.5  # 50% profit
            
            # Check IV expansion
            iv_increasing = iv_change > 0.1  # IV increased by 10%
            
            # Check technical resistance
            near_resistance = current_position.get("near_resistance", False)
            
            if profit_target_met and (iv_increasing or near_resistance):
                return (True, "Profit target met with IV expansion or technical resistance", "equity_to_options")
        
        # Check options to equity conversion
        elif "option" in position_type:
            options_to_equity = pathways.get("options_to_equity", {})
            trigger_conditions = options_to_equity.get("trigger_conditions", [])
            
            # Check IV contraction
            iv_contracting = iv_change < -0.1  # IV decreased by 10%
            
            # Check DTE
            low_dte = days_to_expiration <= 21
            
            # Check for consolidation
            in_consolidation = current_position.get("in_consolidation", False)
            
            if low_dte and (iv_contracting or in_consolidation):
                return (True, "Options approaching expiration with IV contraction or consolidation", "options_to_equity")
        
        return (False, "No conversion triggers met", None)
    
    def generate_conversion_plan(self, current_position, conversion_type, market_indicators):
        """
        Generate a plan for converting between equity and options strategies.
        
        Args:
            current_position (dict): Current position details
            conversion_type (str): Type of conversion (equity_to_options or options_to_equity)
            market_indicators (dict): Current market indicators
            
        Returns:
            dict: Conversion plan
        """
        # Get conversion pathways
        pathways = self.get_integration_framework("conversion_pathways")
        conversion_rules = pathways.get(conversion_type, {}).get("conversion_rules", [])
        
        # Default plan
        plan = {
            "original_position": current_position,
            "conversion_type": conversion_type,
            "actions": [],
            "target_position": {}
        }
        
        symbol = current_position.get("symbol", "")
        quantity = current_position.get("quantity", 0)
        current_price = current_position.get("current_price", 0)
        
        # Handle equity to options conversion
        if conversion_type == "equity_to_options":
            # Calculate approximate delta value of equity position
            equity_delta = quantity * 100  # Each 100 shares ≈ 1.0 delta
            
            # Determine options strategy based on market condition
            market_condition = "sideways"
            iv_rank = 50
            
            if self.context_engine:
                market_indicators = self.context_engine.get_market_indicators()
                market_condition = self.strategy_selector.get_market_condition(market_indicators)
                iv_rank = market_indicators.get("iv_rank", 50)
            
            # Choose appropriate strategy based on IV and market condition
            if market_condition == "bullish":
                if iv_rank > 50:
                    options_strategy = "bull_put_spread"
                    delta_per_contract = 0.30
                else:
                    options_strategy = "bull_call_spread"
                    delta_per_contract = 0.60
            elif market_condition == "bearish":
                if iv_rank > 50:
                    options_strategy = "bear_call_spread"
                    delta_per_contract = 0.30
                else:
                    options_strategy = "bear_put_spread"
                    delta_per_contract = 0.60
            else:  # Sideways
                options_strategy = "iron_condor"
                delta_per_contract = 0.15
            
            # Calculate contracts needed to maintain approximately 50% of delta exposure
            target_delta = equity_delta * 0.5
            contracts_needed = max(1, int(target_delta / (delta_per_contract * 100)))
            
            plan["actions"] = [
                f"Sell {quantity} shares of {symbol} at ~${current_price}",
                f"Replace with {contracts_needed} contracts of {options_strategy} on {symbol}"
            ]
            
            plan["target_position"] = {
                "strategy": options_strategy,
                "symbol": symbol,
                "quantity": contracts_needed,
                "delta_exposure": target_delta,
                "capital_reduction": f"{int((equity_delta - target_delta) / equity_delta * 100)}%"
            }
            
        # Handle options to equity conversion
        elif conversion_type == "options_to_equity":
            # Get options details
            options_delta = current_position.get("delta", 0.5) * 100 * quantity
            
            # Calculate shares to maintain approximately 75% of delta exposure
            target_delta = options_delta * 0.75
            shares_needed = max(1, int(target_delta / 100))
            
            plan["actions"] = [
                f"Close {quantity} contracts of {current_position.get('strategy', 'options')} on {symbol}",
                f"Purchase {shares_needed} shares of {symbol} at ~${current_price}"
            ]
            
            # Check if covered calls would be appropriate
            if market_indicators.get("iv_rank", 0) > 40:
                plan["actions"].append(f"Consider selling covered calls against {shares_needed} shares for additional income")
            
            plan["target_position"] = {
                "strategy": "equity" if market_indicators.get("iv_rank", 0) <= 40 else "covered_call",
                "symbol": symbol,
                "quantity": shares_needed,
                "delta_exposure": target_delta
            }
        
        return plan 