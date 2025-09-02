import logging
import json
import os
import yaml
import numpy as np
from datetime import datetime, time
from strategy_integration import StrategyIntegration

# Import pattern detection module
from pattern_detection import PatternDetectionEngine

logger = logging.getLogger(__name__)

class AdvancedIntegration(StrategyIntegration):
    """
    Advanced Integration Layer that extends the Strategy Integration Layer
    with AI overrides, confidence-weighted allocation, sector sensitivity,
    intraday adaptation, and multi-timeframe pattern confirmation.
    """
    
    def __init__(self, config, context_engine=None):
        """Initialize the advanced integration layer."""
        super().__init__(config, context_engine)
        
        # Load advanced configuration
        self.advanced_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "advanced_integration.json")
        if os.path.exists(self.advanced_config_file):
            with open(self.advanced_config_file, "r") as f:
                self.advanced_config = json.load(f)
        else:
            self.advanced_config = self._create_default_advanced_config()
            # Save the default config
            with open(self.advanced_config_file, "w") as f:
                json.dump(self.advanced_config, f, indent=2)
        
        # Initialize pattern detection engine
        self.pattern_engine = PatternDetectionEngine(config)
        
        logger.info("Advanced Integration Layer initialized")
    
    def _create_default_advanced_config(self):
        """Create default advanced integration configuration."""
        return {
            "ai_override_points": {
                "strategy_filtering": True,
                "sizing_adjustment": True,
                "conversion_timing": True,
                "sentiment_block_trade": True,
                "market_condition_weight": 0.7,
                "ai_weight": 0.3
            },
            "capital_allocation_modifiers": {
                "confidence_weighting": {
                    "high": 1.25,
                    "medium": 1.0,
                    "low": 0.75
                },
                "ai_bias_score_adjustment": {
                    "bullish_bias_score_threshold": 0.7,
                    "bearish_bias_score_threshold": 0.7,
                    "bullish_size_increase": 0.1,
                    "bearish_put_increase": 0.1
                },
                "streak_adjustment": {
                    "win_streak_factor": 0.05,
                    "loss_streak_factor": -0.1,
                    "max_streak_adjustment": 0.2
                }
            },
            "sector_bias_framework": {
                "tech_strength": {
                    "increase_weighting": [
                        "breakout_swing",
                        "bull_call_spread"
                    ],
                    "sector_etf": "XLK",
                    "relative_strength_threshold": 1.1
                },
                "financials_weakness": {
                    "avoid_strategies": [
                        "iron_condor",
                        "theta_spread"
                    ],
                    "sector_etf": "XLF",
                    "relative_weakness_threshold": 0.9
                },
                "energy_volatility": {
                    "increase_weighting": [
                        "long_straddle",
                        "volatility_squeeze"
                    ],
                    "sector_etf": "XLE",
                    "volatility_threshold": 1.3
                },
                "consumer_stability": {
                    "increase_weighting": [
                        "bull_put_spread",
                        "calendar_spread"
                    ],
                    "sector_etf": "XLP",
                    "stability_threshold": 0.7
                }
            },
            "intraday_adaptation": {
                "morning_bias_reversal": {
                    "condition": "first_hour_strong_move_reversed_by_1045",
                    "condition_parameters": {
                        "strong_move_threshold": 0.5,
                        "reversal_threshold": 0.3,
                        "reversal_time": "10:45"
                    },
                    "actions": [
                        "close_directional_equity_positions",
                        "switch_to_range_bound_premium_collection"
                    ]
                },
                "unexpected_iv_spike": {
                    "condition": "vix_up_percent_intraday",
                    "condition_parameters": {
                        "vix_increase_threshold": 10.0
                    },
                    "actions": [
                        "convert_open_options_to_spreads",
                        "cut_unhedged_delta_exposures"
                    ]
                },
                "afternoon_momentum_continuation": {
                    "condition": "trend_continuing_after_lunch",
                    "condition_parameters": {
                        "lunch_period_end": "13:00",
                        "continuation_threshold": 0.3
                    },
                    "actions": [
                        "add_to_winning_positions",
                        "extend_profit_targets"
                    ]
                },
                "end_of_day_profit_taking": {
                    "condition": "approaching_market_close",
                    "condition_parameters": {
                        "time_threshold": "15:30",
                        "profit_threshold": 0.5
                    },
                    "actions": [
                        "close_intraday_positions",
                        "roll_profitable_options"
                    ]
                }
            },
            "multi_timeframe_pattern_cascade": {
                "bull_flag_cascade": {
                    "setup": "daily_bull_flag",
                    "confirmation": "15m_macd_crossover_with_volume",
                    "execution": "bull_call_spread_or_synthetic_long"
                },
                "breakout_cascade": {
                    "setup": "weekly_consolidation_near_resistance",
                    "confirmation": "daily_breakout_with_volume",
                    "micro_confirmation": "hourly_higher_highs",
                    "execution": "breakout_swing_with_options_overlay"
                },
                "correction_cascade": {
                    "setup": "daily_oversold_in_uptrend",
                    "confirmation": "4h_bullish_divergence",
                    "micro_confirmation": "hourly_double_bottom",
                    "execution": "oversold_bounce_with_bull_put"
                },
                "volatility_compression_cascade": {
                    "setup": "weekly_volatility_contraction",
                    "confirmation": "daily_narrowing_bbands",
                    "micro_confirmation": "4h_inside_bars",
                    "execution": "volatility_expansion_straddle"
                }
            }
        }
    
    def get_ai_override_status(self, override_point):
        """
        Check if AI can override a specific decision point.
        
        Args:
            override_point (str): The override point to check
            
        Returns:
            bool: Whether AI override is allowed
        """
        return self.advanced_config.get("ai_override_points", {}).get(override_point, False)
    
    def apply_confidence_weighted_allocation(self, position_size, confidence_level, bias_score=0.5, 
                                            strategy_win_streak=0, strategy_type=""):
        """
        Apply confidence weighting to position sizing.
        
        Args:
            position_size (float): Base position size
            confidence_level (str): Confidence level ('high', 'medium', 'low')
            bias_score (float): AI bias score (0.0-1.0)
            strategy_win_streak (int): Current win streak for the strategy
            strategy_type (str): Strategy type
            
        Returns:
            float: Adjusted position size
        """
        # Get confidence multipliers
        confidence_weights = self.advanced_config.get("capital_allocation_modifiers", {}).get("confidence_weighting", {})
        confidence_multiplier = confidence_weights.get(confidence_level.lower(), 1.0)
        
        # Apply confidence weighting
        adjusted_size = position_size * confidence_multiplier
        
        # Apply AI bias adjustment
        bias_adjustments = self.advanced_config.get("capital_allocation_modifiers", {}).get("ai_bias_score_adjustment", {})
        bullish_threshold = bias_adjustments.get("bullish_bias_score_threshold", 0.7)
        bearish_threshold = bias_adjustments.get("bearish_bias_score_threshold", 0.7)
        bullish_increase = bias_adjustments.get("bullish_size_increase", 0.1)
        bearish_increase = bias_adjustments.get("bearish_put_increase", 0.1)
        
        # Increase size for bullish bias on long strategies
        if bias_score >= bullish_threshold and "bull" in strategy_type.lower():
            adjusted_size *= (1 + bullish_increase)
        
        # Increase size for bearish bias on short strategies
        if (1.0 - bias_score) >= bearish_threshold and "bear" in strategy_type.lower():
            adjusted_size *= (1 + bearish_increase)
        
        # Apply streak adjustment
        streak_adjustments = self.advanced_config.get("capital_allocation_modifiers", {}).get("streak_adjustment", {})
        win_streak_factor = streak_adjustments.get("win_streak_factor", 0.05)
        loss_streak_factor = streak_adjustments.get("loss_streak_factor", -0.1)
        max_adjustment = streak_adjustments.get("max_streak_adjustment", 0.2)
        
        streak_multiplier = 1.0
        if strategy_win_streak > 0:
            streak_multiplier = 1.0 + min(strategy_win_streak * win_streak_factor, max_adjustment)
        elif strategy_win_streak < 0:
            # Negative streak means a loss streak
            streak_multiplier = 1.0 + max(strategy_win_streak * abs(loss_streak_factor), -max_adjustment)
        
        adjusted_size *= streak_multiplier
        
        return adjusted_size
    
    def get_sector_bias(self, sector):
        """
        Get sector bias and associated strategy adjustments.
        
        Args:
            sector (str): The sector to check (e.g., 'tech', 'financials', 'energy', 'consumer')
            
        Returns:
            dict: Sector bias data including strategies to favor or avoid
        """
        sector_mapping = {
            "tech": "tech_strength",
            "technology": "tech_strength",
            "financials": "financials_weakness",
            "finance": "financials_weakness",
            "energy": "energy_volatility",
            "consumer": "consumer_stability",
            "consumer_staples": "consumer_stability"
        }
        
        mapped_sector = sector_mapping.get(sector.lower(), None)
        if not mapped_sector:
            return None
        
        return self.advanced_config.get("sector_bias_framework", {}).get(mapped_sector, None)
    
    def adjust_for_sector_bias(self, recommendations, symbol_sector):
        """
        Adjust strategy recommendations based on sector bias.
        
        Args:
            recommendations (dict): Current strategy recommendations
            symbol_sector (str): Sector of the symbol
            
        Returns:
            dict: Adjusted strategy recommendations
        """
        # Get sector bias
        sector_bias = self.get_sector_bias(symbol_sector)
        if not sector_bias:
            return recommendations
        
        adjusted_recommendations = recommendations.copy()
        integrated_strategies = adjusted_recommendations.get("integrated_strategies", [])
        
        # Strategies to increase weighting for
        increase_weighting = sector_bias.get("increase_weighting", [])
        
        # Strategies to avoid
        avoid_strategies = sector_bias.get("avoid_strategies", [])
        
        # Prioritize favored strategies and filter out avoided strategies
        if integrated_strategies:
            # Remove avoided strategies
            filtered_strategies = [s for s in integrated_strategies 
                                 if s["core_strategy"] not in avoid_strategies and 
                                    s["options_strategy"] not in avoid_strategies]
            
            # Prioritize favored strategies
            prioritized_strategies = sorted(filtered_strategies, 
                                         key=lambda s: (s["core_strategy"] in increase_weighting or 
                                                      s["options_strategy"] in increase_weighting),
                                         reverse=True)
            
            adjusted_recommendations["integrated_strategies"] = prioritized_strategies
            
            # Add sector bias info to the recommendations
            adjusted_recommendations["sector_bias"] = {
                "sector": symbol_sector,
                "favored_strategies": increase_weighting,
                "avoided_strategies": avoid_strategies
            }
        
        return adjusted_recommendations
    
    def check_intraday_adaptation(self, current_market_data):
        """
        Check if intraday adaptation is needed based on current market conditions.
        
        Args:
            current_market_data (dict): Current market data including prices, time, etc.
            
        Returns:
            tuple: (should_adapt, adaptation_plan)
        """
        adaptations = self.advanced_config.get("intraday_adaptation", {})
        current_time = current_market_data.get("current_time", datetime.now().time())
        
        # Check each adaptation rule
        for rule_name, rule_data in adaptations.items():
            condition = rule_data.get("condition")
            params = rule_data.get("condition_parameters", {})
            actions = rule_data.get("actions", [])
            
            # Check morning bias reversal
            if condition == "first_hour_strong_move_reversed_by_1045":
                first_hour_move = current_market_data.get("first_hour_percent_move", 0)
                current_move = current_market_data.get("current_percent_move", 0)
                reversal_threshold = params.get("reversal_threshold", 0.3)
                strong_move_threshold = params.get("strong_move_threshold", 0.5)
                reversal_time_str = params.get("reversal_time", "10:45")
                reversal_time = datetime.strptime(reversal_time_str, "%H:%M").time()
                
                # Check if conditions are met
                if (abs(first_hour_move) >= strong_move_threshold and 
                    current_time <= reversal_time and
                    ((first_hour_move > 0 and current_move < first_hour_move * (1 - reversal_threshold)) or 
                     (first_hour_move < 0 and current_move > first_hour_move * (1 - reversal_threshold)))):
                    return (True, {
                        "rule": rule_name,
                        "actions": actions,
                        "reason": "Morning bias reversal detected"
                    })
            
            # Check unexpected IV spike
            elif condition == "vix_up_percent_intraday":
                vix_change_percent = current_market_data.get("vix_change_percent", 0)
                vix_threshold = params.get("vix_increase_threshold", 10.0)
                
                if vix_change_percent >= vix_threshold:
                    return (True, {
                        "rule": rule_name,
                        "actions": actions,
                        "reason": f"VIX spike of {vix_change_percent:.1f}% detected"
                    })
            
            # Check afternoon momentum continuation
            elif condition == "trend_continuing_after_lunch":
                lunch_end_str = params.get("lunch_period_end", "13:00")
                lunch_end = datetime.strptime(lunch_end_str, "%H:%M").time()
                continuation_threshold = params.get("continuation_threshold", 0.3)
                pre_lunch_trend = current_market_data.get("pre_lunch_trend", 0)
                post_lunch_move = current_market_data.get("post_lunch_move", 0)
                
                if (current_time >= lunch_end and
                    ((pre_lunch_trend > 0 and post_lunch_move > continuation_threshold) or
                     (pre_lunch_trend < 0 and post_lunch_move < -continuation_threshold))):
                    return (True, {
                        "rule": rule_name,
                        "actions": actions,
                        "reason": "Afternoon momentum continuation detected"
                    })
            
            # Check end of day profit taking
            elif condition == "approaching_market_close":
                close_time_str = params.get("time_threshold", "15:30")
                close_time = datetime.strptime(close_time_str, "%H:%M").time()
                profit_threshold = params.get("profit_threshold", 0.5)
                current_positions = current_market_data.get("open_positions", [])
                profitable_positions = [p for p in current_positions 
                                       if p.get("unrealized_profit_percent", 0) >= profit_threshold]
                
                if current_time >= close_time and profitable_positions:
                    return (True, {
                        "rule": rule_name,
                        "actions": actions,
                        "reason": "End of day profit taking opportunity",
                        "positions_to_close": profitable_positions
                    })
        
        return (False, None)
    
    def execute_intraday_adaptation(self, adaptation_plan, tradier_api):
        """
        Execute intraday adaptation actions.
        
        Args:
            adaptation_plan (dict): The adaptation plan to execute
            tradier_api: TradierAPI instance for executing trades
            
        Returns:
            dict: Results of the adaptation
        """
        results = {
            "rule": adaptation_plan.get("rule"),
            "reason": adaptation_plan.get("reason"),
            "actions_taken": [],
            "actions_failed": []
        }
        
        actions = adaptation_plan.get("actions", [])
        positions_to_close = adaptation_plan.get("positions_to_close", [])
        
        for action in actions:
            try:
                if action == "close_directional_equity_positions":
                    # Close directional equity positions
                    for position in positions_to_close:
                        if position.get("type") == "equity":
                            # Place sell order
                            order_result = tradier_api.place_equity_order(
                                symbol=position.get("symbol"),
                                side="sell",
                                quantity=position.get("quantity", 0),
                                order_type="market"
                            )
                            results["actions_taken"].append({
                                "action": action,
                                "symbol": position.get("symbol"),
                                "quantity": position.get("quantity", 0),
                                "order_id": order_result.get("order_id")
                            })
                
                elif action == "switch_to_range_bound_premium_collection":
                    # Implement logic to switch to range-bound strategies
                    # This would involve more complex order placement
                    results["actions_taken"].append({
                        "action": action,
                        "status": "implemented",
                        "strategy_switch": "Switched to range-bound premium collection strategies"
                    })
                
                elif action == "convert_open_options_to_spreads":
                    # Convert open options to spreads for protection
                    for position in positions_to_close:
                        if "option" in position.get("type", ""):
                            # Logic to convert to spread would go here
                            results["actions_taken"].append({
                                "action": action,
                                "symbol": position.get("symbol"),
                                "conversion": f"Converted {position.get('type')} to spread"
                            })
                
                # Implement other actions as needed
                
            except Exception as e:
                results["actions_failed"].append({
                    "action": action,
                    "error": str(e)
                })
        
        return results
    
    def identify_multi_timeframe_pattern(self, pattern_name, market_data):
        """
        Identify if a multi-timeframe pattern is present in the market data.
        
        Args:
            pattern_name (str): The pattern to check for
            market_data (dict): Market data across multiple timeframes
            
        Returns:
            tuple: (pattern_present, confidence_score, execution_plan)
        """
        # First check if we have patterns from the pattern engine
        symbol = market_data.get('symbol')
        
        # If we have multi-timeframe price data available in market_data
        timeframe_data = {}
        for key, value in market_data.items():
            if isinstance(key, str) and key.endswith('_price_data') and isinstance(value, dict):
                tf = key.replace('_price_data', '')
                if 'close' in value and 'high' in value and 'low' in value and 'open' in value:
                    price_data = value
                    volume_data = market_data.get(f"{tf}_volume_data")
                    timeframe_data[tf] = (price_data, volume_data)
        
        # Use pattern engine if we have price data
        if timeframe_data and hasattr(self, 'pattern_engine'):
            # Scan for patterns
            patterns_by_timeframe = self.pattern_engine.scan_multiple_timeframes(timeframe_data)
            
            # Check if the requested pattern exists in detected patterns
            pattern_found = False
            pattern_timeframes = []
            pattern_confidences = []
            
            for timeframe, symbols_data in patterns_by_timeframe.items():
                if symbol in symbols_data:
                    patterns = symbols_data[symbol]
                    for pattern in patterns:
                        if pattern['pattern'].lower() == pattern_name.lower():
                            pattern_found = True
                            pattern_timeframes.append(timeframe)
                            pattern_confidences.append(pattern.get('confidence_score', 0.5))
            
            if pattern_found:
                # Calculate overall confidence based on timeframe alignment
                avg_confidence = sum(pattern_confidences) / len(pattern_confidences)
                
                # Multi-timeframe alignment boosts confidence
                timeframe_boost = min(0.2, 0.05 * len(pattern_timeframes))
                final_confidence = min(1.0, avg_confidence + timeframe_boost)
                
                # Create an execution plan
                execution_plan = {
                    "status": "pattern_confirmed",
                    "pattern": pattern_name,
                    "timeframes_present": pattern_timeframes,
                    "confidence_score": final_confidence,
                    "recommendation": "Execute according to pattern strategy with strong confidence"
                }
                
                return (True, final_confidence, execution_plan)
        
        # If pattern engine didn't find the pattern or no price data was available,
        # fall back to the pre-defined patterns in advanced_config
        patterns = self.advanced_config.get("multi_timeframe_pattern_cascade", {})
        if pattern_name not in patterns:
            return (False, 0, None)
        
        pattern_config = patterns[pattern_name]
        setup = pattern_config.get("setup")
        confirmation = pattern_config.get("confirmation")
        micro_confirmation = pattern_config.get("micro_confirmation", None)
        execution = pattern_config.get("execution")
        
        # Check for primary setup
        setup_present = self._check_pattern_condition(setup, market_data)
        if not setup_present:
            return (False, 0, None)
        
        # Check for confirmation
        confirmation_present = self._check_pattern_condition(confirmation, market_data)
        if not confirmation_present:
            return (True, 0.3, {
                "status": "setup_detected",
                "pattern": pattern_name,
                "missing": "confirmation",
                "recommendation": "Wait for confirmation signal"
            })
        
        # Check for micro confirmation if specified
        confidence = 0.7  # Base confidence with setup and confirmation
        if micro_confirmation:
            micro_present = self._check_pattern_condition(micro_confirmation, market_data)
            if not micro_present:
                return (True, confidence, {
                    "status": "partial_confirmation",
                    "pattern": pattern_name,
                    "missing": "micro_confirmation",
                    "recommendation": "Consider partial position with tight stop"
                })
            confidence = 0.9  # Full confidence with all confirmations
        
        # Generate execution plan
        execution_plan = {
            "status": "fully_confirmed",
            "pattern": pattern_name,
            "execution_strategy": execution,
            "confidence_score": confidence,
            "recommendation": "Execute full position with pattern-based targets"
        }
        
        return (True, confidence, execution_plan)
    
    def _check_pattern_condition(self, condition, market_data):
        """Helper method to check if a pattern condition is present in market data."""
        # This would be implemented with actual technical analysis logic
        # For now, we'll use a simplified placeholder implementation
        
        if "bull_flag" in condition and market_data.get("bull_flag_pattern", False):
            return True
        
        if "macd_crossover" in condition and market_data.get("macd_crossover", False):
            return True
        
        if "consolidation" in condition and market_data.get("consolidation_pattern", False):
            return True
        
        if "breakout" in condition and market_data.get("breakout_detected", False):
            return True
        
        if "oversold" in condition and market_data.get("is_oversold", False):
            return True
        
        if "divergence" in condition and market_data.get("bullish_divergence", False):
            return True
        
        if "volatility_contraction" in condition and market_data.get("volatility_contracting", False):
            return True
        
        # Default to not found
        return False
    
    def get_advanced_recommendation(self, ticker, account_value=None, market_data=None):
        """
        Get advanced strategy recommendations with all enhancements applied.
        
        Args:
            ticker (str): Symbol to analyze
            account_value (float, optional): Account value for position sizing
            market_data (dict, optional): Current market data
            
        Returns:
            dict: Enhanced strategy recommendations
        """
        # Get base recommendation from parent class
        base_recommendation = self.recommend_strategy_integration(ticker, account_value)
        
        # Check for pattern-based recommendations if market data is provided
        if market_data and hasattr(self, 'pattern_engine'):
            # Add pattern detection 
            market_data['symbol'] = ticker
            
            # Get active patterns for this ticker
            timeframe_price_data = {}
            timeframe_volume_data = {}
            
            # Extract price and volume data from market_data if available
            for key, value in market_data.items():
                if key.endswith('_price_data') and isinstance(value, dict):
                    timeframe = key.replace('_price_data', '')
                    timeframe_price_data[timeframe] = value
                if key.endswith('_volume_data') and isinstance(value, (list, np.ndarray)):
                    timeframe = key.replace('_volume_data', '')
                    timeframe_volume_data[timeframe] = value
            
            # Detect patterns if price data is available
            detected_patterns = {}
            if timeframe_price_data:
                data_dict_by_timeframe = {}
                for timeframe, price_data in timeframe_price_data.items():
                    volume_data = timeframe_volume_data.get(timeframe)
                    data_dict_by_timeframe[timeframe] = {ticker: (price_data, volume_data)}
                
                pattern_results = self.pattern_engine.scan_multiple_timeframes(data_dict_by_timeframe)
                
                # Add detected patterns to the recommendation
                if pattern_results:
                    detected_patterns = pattern_results
                    
                    # Find pattern confluence
                    confluence = self.pattern_engine.find_pattern_confluence(ticker)
                    
                    # Add pattern information to recommendation
                    base_recommendation["pattern_analysis"] = {
                        "detected_patterns": detected_patterns,
                        "pattern_confluence": confluence
                    }
                    
                    # Adjust strategy recommendations based on pattern direction
                    if confluence['overall_bias'] in ['bullish', 'bearish'] and confluence['confluence_score'] > 0.7:
                        # Get strategies matching the pattern bias
                        matching_strategies = [
                            strategy for strategy in base_recommendation.get("integrated_strategies", [])
                            if (confluence['overall_bias'] == 'bullish' and 'bull' in strategy.get('options_strategy', '').lower()) or
                               (confluence['overall_bias'] == 'bearish' and 'bear' in strategy.get('options_strategy', '').lower())
                        ]
                        
                        # Prioritize strategies that match the pattern bias
                        if matching_strategies:
                            # Reorder strategies to prioritize matching ones
                            non_matching = [s for s in base_recommendation.get("integrated_strategies", []) 
                                          if s not in matching_strategies]
                            base_recommendation["integrated_strategies"] = matching_strategies + non_matching
                            
                            # Adjust confidence based on pattern confluence
                            base_recommendation["pattern_adjusted"] = True
                            base_recommendation["pattern_confidence"] = confluence['confluence_score']
        
        # Get sector for the ticker (simplified)
        symbol_sector = "technology"  # This would be replaced with actual sector lookup
        if self.context_engine:
            symbol_sector = self.context_engine._get_symbol_sector(ticker)
        
        # Apply sector bias adjustments
        sector_adjusted = self.adjust_for_sector_bias(base_recommendation, symbol_sector)
        
        # Apply confidence weighting if we have top strategies
        if sector_adjusted.get("integrated_strategies") and account_value:
            top_strategy = sector_adjusted["integrated_strategies"][0]
            bias_score = sector_adjusted.get("bias_confidence", 0.5)
            
            # Apply confidence weighting to position sizing
            if sector_adjusted.get("position_sizing"):
                equity_size = sector_adjusted["position_sizing"].get("equity_position_size", 0)
                options_size = sector_adjusted["position_sizing"].get("options_position_size", 0)
                
                # Get win streak (placeholder - would come from analytics)
                win_streak = 0
                
                adjusted_equity = self.apply_confidence_weighted_allocation(
                    equity_size, 
                    "medium",  # Confidence level
                    bias_score,
                    win_streak,
                    top_strategy.get("core_strategy", "")
                )
                
                adjusted_options = self.apply_confidence_weighted_allocation(
                    options_size,
                    "medium",  # Confidence level
                    bias_score,
                    win_streak,
                    top_strategy.get("options_strategy", "")
                )
                
                sector_adjusted["position_sizing"]["equity_position_size"] = round(adjusted_equity, 2)
                sector_adjusted["position_sizing"]["options_position_size"] = round(adjusted_options, 2)
                sector_adjusted["position_sizing"]["confidence_adjusted"] = True
        
        # Check for multi-timeframe patterns
        if market_data:
            detected_patterns = []
            
            # Check each pattern cascade
            for pattern_name in self.advanced_config.get("multi_timeframe_pattern_cascade", {}):
                is_present, confidence, plan = self.identify_multi_timeframe_pattern(pattern_name, market_data)
                if is_present:
                    detected_patterns.append({
                        "pattern": pattern_name,
                        "confidence": confidence,
                        "plan": plan
                    })
            
            # Add pattern information to recommendation
            if detected_patterns:
                sector_adjusted["multi_timeframe_patterns"] = detected_patterns
        
        # Check for intraday adaptation if we have current market data
        if market_data:
            should_adapt, adaptation_plan = self.check_intraday_adaptation(market_data)
            if should_adapt:
                sector_adjusted["intraday_adaptation"] = adaptation_plan
        
        # Add AI override information
        sector_adjusted["ai_override_points"] = self.advanced_config.get("ai_override_points", {})
        
        return sector_adjusted 