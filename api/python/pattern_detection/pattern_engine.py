import pandas as pd
import numpy as np
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from .pattern_definitions import get_all_patterns

logger = logging.getLogger(__name__)

class PatternDetectionEngine:
    """
    Engine for detecting chart patterns across multiple timeframes and instruments.
    Coordinates pattern detection, filtering, and integration with trading strategies.
    """
    
    def __init__(self, config=None):
        """Initialize the pattern detection engine."""
        self.config = config or {}
        self.patterns = get_all_patterns()
        self.scanning_active = False
        self.detected_patterns = {}  # Store detected patterns by symbol
        self.pattern_history = {}    # Store historical pattern performance
        
        # Configure detection parameters
        self.detection_params = {
            "max_lookback_bars": 200,  # Maximum number of bars to analyze for patterns
            "min_pattern_confidence": 0.6,  # Minimum confidence score to report patterns
            "parallel_processing": True,  # Use parallel processing for scanning
            "max_workers": 4,  # Maximum number of worker threads
            "volume_required": True,  # Require volume data for pattern detection
            "min_liquidity": 100000,  # Minimum average daily volume
        }
        
        # Update parameters from config if provided
        if config and "pattern_detection" in config:
            self.detection_params.update(config["pattern_detection"])
            
        logger.info(f"Pattern Detection Engine initialized with {len(self.patterns)} patterns")
    
    def detect_patterns(self, price_data, volume_data=None, symbol=None, timeframe=None):
        """
        Detect patterns in a single instrument's price data.
        
        Args:
            price_data (pd.DataFrame): OHLC price data
            volume_data (np.array, optional): Volume data
            symbol (str, optional): Symbol identifier
            timeframe (str, optional): Timeframe of the data
            
        Returns:
            list: Detected patterns with metadata
        """
        if price_data is None or len(price_data) < 30:
            logger.warning(f"Insufficient data for pattern detection: {symbol or 'Unknown'}")
            return []
            
        # Create a copy of the data limited to max lookback
        price_data = price_data.iloc[-self.detection_params["max_lookback_bars"]:].copy()
        
        # Use volume data if provided and required
        has_volume = volume_data is not None and len(volume_data) >= len(price_data)
        if self.detection_params["volume_required"] and not has_volume:
            logger.warning(f"Volume data required but not provided for {symbol or 'Unknown'}")
            return []
            
        # If no volume data provided, use empty array
        if volume_data is None:
            volume_data = np.zeros(len(price_data))
        
        # Check liquidity (average volume)
        if has_volume and np.mean(volume_data) < self.detection_params["min_liquidity"]:
            logger.debug(f"Insufficient liquidity for {symbol or 'Unknown'}")
            return []
        
        all_detected_patterns = []
        
        # Run detection for each pattern type
        for pattern in self.patterns:
            try:
                start_time = time.time()
                detected = pattern.detect(price_data, volume_data)
                
                # Filter patterns by confidence score
                if detected:
                    for p in detected:
                        confidence = p.get('reliability_score', 0)
                        
                        # Add additional context to detected patterns
                        p['symbol'] = symbol
                        p['timeframe'] = timeframe
                        p['detection_time'] = pd.Timestamp.now().isoformat()
                        
                        # Apply pattern validation
                        if hasattr(pattern, 'validate_pattern'):
                            is_valid, adjusted_confidence, validation_notes = pattern.validate_pattern(p)
                            p['confidence_score'] = adjusted_confidence
                            p['validation_notes'] = validation_notes
                            p['is_valid'] = is_valid
                            
                            if is_valid and adjusted_confidence >= self.detection_params["min_pattern_confidence"]:
                                all_detected_patterns.append(p)
                        else:
                            # If no validation method, use reliability score as confidence
                            p['confidence_score'] = confidence
                            if confidence >= self.detection_params["min_pattern_confidence"]:
                                all_detected_patterns.append(p)
                
                elapsed = time.time() - start_time
                logger.debug(f"Detected {len(detected)} {pattern.name} patterns for {symbol or 'Unknown'} in {elapsed:.3f}s")
                
            except Exception as e:
                logger.error(f"Error detecting {pattern.name} patterns: {str(e)}")
        
        return all_detected_patterns
    
    def scan_multiple_symbols(self, data_dict, timeframe=None):
        """
        Scan multiple symbols for patterns in parallel.
        
        Args:
            data_dict (dict): Dictionary mapping symbols to (price_data, volume_data) tuples
            timeframe (str, optional): Timeframe of the data
            
        Returns:
            dict: Dictionary of detected patterns by symbol
        """
        results = {}
        
        if self.detection_params["parallel_processing"]:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.detection_params["max_workers"]) as executor:
                future_to_symbol = {}
                
                for symbol, (price_data, volume_data) in data_dict.items():
                    future = executor.submit(
                        self.detect_patterns, 
                        price_data, 
                        volume_data, 
                        symbol, 
                        timeframe
                    )
                    future_to_symbol[future] = symbol
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        patterns = future.result()
                        if patterns:
                            results[symbol] = patterns
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")
        else:
            # Sequential processing
            for symbol, (price_data, volume_data) in data_dict.items():
                try:
                    patterns = self.detect_patterns(price_data, volume_data, symbol, timeframe)
                    if patterns:
                        results[symbol] = patterns
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
        
        return results
    
    def scan_multiple_timeframes(self, data_dict_by_timeframe):
        """
        Scan multiple timeframes for patterns.
        
        Args:
            data_dict_by_timeframe (dict): Dictionary mapping timeframes to symbol data dictionaries
            
        Returns:
            dict: Nested dictionary of detected patterns by timeframe and symbol
        """
        results = {}
        
        for timeframe, data_dict in data_dict_by_timeframe.items():
            timeframe_results = self.scan_multiple_symbols(data_dict, timeframe)
            if timeframe_results:
                results[timeframe] = timeframe_results
        
        return results
    
    def start_continuous_scanning(self, data_provider, symbols, timeframes, scan_interval=60):
        """
        Start continuous scanning for patterns.
        
        Args:
            data_provider: Function or object that provides price and volume data
            symbols (list): List of symbols to scan
            timeframes (list): List of timeframes to scan
            scan_interval (int): Interval between scans in seconds
        """
        self.scanning_active = True
        logger.info(f"Starting continuous pattern scanning for {len(symbols)} symbols on {len(timeframes)} timeframes")
        
        while self.scanning_active:
            try:
                start_time = time.time()
                
                # Create data dictionary by timeframe
                data_dict_by_timeframe = {}
                for timeframe in timeframes:
                    data_dict_by_timeframe[timeframe] = {}
                    for symbol in symbols:
                        try:
                            price_data, volume_data = data_provider.get_data(symbol, timeframe)
                            data_dict_by_timeframe[timeframe][symbol] = (price_data, volume_data)
                        except Exception as e:
                            logger.error(f"Error getting data for {symbol} ({timeframe}): {str(e)}")
                
                # Scan for patterns
                results = self.scan_multiple_timeframes(data_dict_by_timeframe)
                
                # Store results
                self.update_detected_patterns(results)
                
                # Log a summary
                total_patterns = sum(len(patterns) for timeframe_data in results.values() 
                                    for patterns in timeframe_data.values())
                logger.info(f"Detected {total_patterns} patterns across {len(results)} timeframes")
                
                # Sleep until next scan
                elapsed = time.time() - start_time
                sleep_time = max(0, scan_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Error in continuous scanning: {str(e)}")
                time.sleep(5)  # Sleep briefly before retrying
    
    def stop_continuous_scanning(self):
        """Stop continuous scanning."""
        self.scanning_active = False
        logger.info("Stopped continuous pattern scanning")
    
    def update_detected_patterns(self, new_results):
        """
        Update the dictionary of detected patterns.
        
        Args:
            new_results (dict): New pattern detection results
        """
        # First, create a combined structure if needed
        if not self.detected_patterns:
            self.detected_patterns = {}
        
        # Update with new results
        for timeframe, timeframe_data in new_results.items():
            if timeframe not in self.detected_patterns:
                self.detected_patterns[timeframe] = {}
                
            for symbol, patterns in timeframe_data.items():
                # Track patterns by symbol and timeframe
                self.detected_patterns[timeframe][symbol] = patterns
    
    def get_active_patterns(self, symbol=None, timeframe=None, pattern_type=None, direction=None):
        """
        Get active patterns that have been detected.
        
        Args:
            symbol (str, optional): Filter by symbol
            timeframe (str, optional): Filter by timeframe
            pattern_type (str, optional): Filter by pattern type (continuation/reversal)
            direction (str, optional): Filter by direction (bullish/bearish)
            
        Returns:
            list: Filtered list of active patterns
        """
        active_patterns = []
        
        # Filter by timeframe if specified
        timeframes = [timeframe] if timeframe else self.detected_patterns.keys()
        
        for tf in timeframes:
            if tf not in self.detected_patterns:
                continue
                
            # Filter by symbol if specified
            symbols = [symbol] if symbol else self.detected_patterns[tf].keys()
            
            for sym in symbols:
                if sym not in self.detected_patterns[tf]:
                    continue
                    
                # Apply additional filters
                for pattern in self.detected_patterns[tf][sym]:
                    # Filter by pattern type
                    if pattern_type and pattern.get('pattern_type') != pattern_type:
                        continue
                        
                    # Filter by direction
                    if direction and pattern.get('direction') != direction:
                        continue
                        
                    active_patterns.append(pattern)
        
        return active_patterns
    
    def find_pattern_confluence(self, symbol, primary_timeframe=None):
        """
        Find pattern confluence across multiple timeframes for a symbol.
        
        Args:
            symbol (str): Symbol to check
            primary_timeframe (str, optional): Primary timeframe to focus on
            
        Returns:
            dict: Confluence analysis with primary patterns and confirming patterns
        """
        confluence = {
            'symbol': symbol,
            'primary_patterns': [],
            'confirming_patterns': [],
            'overall_bias': 'neutral',
            'confluence_score': 0
        }
        
        # Collect all patterns for this symbol across timeframes
        all_patterns = []
        for timeframe, symbols in self.detected_patterns.items():
            if symbol in symbols:
                for pattern in symbols[symbol]:
                    pattern['timeframe'] = timeframe
                    all_patterns.append(pattern)
        
        if not all_patterns:
            return confluence
        
        # Separate into primary and confirming patterns
        for pattern in all_patterns:
            if primary_timeframe and pattern['timeframe'] == primary_timeframe:
                confluence['primary_patterns'].append(pattern)
            else:
                confluence['confirming_patterns'].append(pattern)
        
        # If no primary timeframe was specified, use the largest timeframe as primary
        if not primary_timeframe and all_patterns:
            timeframe_order = ['monthly', 'weekly', 'daily', '4h', '1h', '30m', '15m', '5m', '1m']
            for tf in timeframe_order:
                primary_candidates = [p for p in all_patterns if p['timeframe'] == tf]
                if primary_candidates:
                    confluence['primary_patterns'] = primary_candidates
                    confirming = [p for p in all_patterns if p not in primary_candidates]
                    confluence['confirming_patterns'] = confirming
                    break
        
        # Calculate overall bias and confluence score
        if confluence['primary_patterns'] or confluence['confirming_patterns']:
            bullish_count = len([p for p in all_patterns if p['direction'] == 'bullish'])
            bearish_count = len([p for p in all_patterns if p['direction'] == 'bearish'])
            
            # Weight by confidence and timeframe
            bullish_score = sum(self._calculate_pattern_weight(p) for p in all_patterns 
                               if p['direction'] == 'bullish')
            bearish_score = sum(self._calculate_pattern_weight(p) for p in all_patterns 
                               if p['direction'] == 'bearish')
            
            # Determine bias
            if bullish_score > bearish_score:
                bias_strength = bullish_score / (bullish_score + bearish_score) if (bullish_score + bearish_score) > 0 else 0.5
                confluence['overall_bias'] = 'bullish'
                confluence['bias_strength'] = bias_strength
            elif bearish_score > bullish_score:
                bias_strength = bearish_score / (bullish_score + bearish_score) if (bullish_score + bearish_score) > 0 else 0.5
                confluence['overall_bias'] = 'bearish'
                confluence['bias_strength'] = bias_strength
            else:
                confluence['overall_bias'] = 'neutral'
                confluence['bias_strength'] = 0.5
            
            # Calculate confluence score (0-1)
            # Higher when multiple aligned patterns exist across timeframes
            timeframe_count = len(set(p['timeframe'] for p in all_patterns))
            pattern_type_count = len(set(p['pattern'] for p in all_patterns))
            
            max_score = 1.0
            base_score = 0.3 + (0.2 * min(timeframe_count / 3, 1.0)) + (0.2 * min(pattern_type_count / 3, 1.0))
            
            # Boost score if patterns align in direction
            if bullish_count > 0 and bearish_count > 0:
                alignment_penalty = 0.3 * min(bullish_count, bearish_count) / max(bullish_count, bearish_count)
                final_score = min(max_score, base_score - alignment_penalty)
            else:
                alignment_boost = 0.3  # All patterns align in direction
                final_score = min(max_score, base_score + alignment_boost)
            
            confluence['confluence_score'] = round(final_score, 2)
        
        return confluence
    
    def _calculate_pattern_weight(self, pattern):
        """
        Calculate weight of a pattern based on confidence and timeframe.
        
        Args:
            pattern (dict): Pattern dictionary
            
        Returns:
            float: Pattern weight for confluence calculations
        """
        # Base weight is confidence score
        weight = pattern.get('confidence_score', pattern.get('reliability_score', 0.5))
        
        # Adjust by timeframe (higher timeframes get more weight)
        timeframe_weights = {
            'monthly': 5.0,
            'weekly': 3.0,
            'daily': 2.0,
            '4h': 1.5,
            '1h': 1.2,
            '30m': 1.0,
            '15m': 0.8,
            '5m': 0.6,
            '1m': 0.4
        }
        
        timeframe_multiplier = timeframe_weights.get(pattern.get('timeframe', '1h'), 1.0)
        return weight * timeframe_multiplier
    
    def update_pattern_tracking(self, symbol, price_data, volume_data=None):
        """
        Update tracking of existing patterns to determine success/failure.
        
        Args:
            symbol (str): Symbol to update
            price_data (pd.DataFrame): Updated price data
            volume_data (np.array, optional): Updated volume data
        """
        if symbol not in self.pattern_history:
            self.pattern_history[symbol] = []
        
        # Check each timeframe for this symbol
        for timeframe, symbols in self.detected_patterns.items():
            if symbol not in symbols:
                continue
                
            # Get patterns for this symbol and timeframe
            patterns = symbols[symbol]
            active_patterns = []
            
            for pattern in patterns:
                # Check if pattern completion index is in the data
                if 'breakout_idx' in pattern:
                    breakout_idx = pattern['breakout_idx']
                    latest_idx = len(price_data) - 1
                    
                    # Determine if pattern has reached target or stop loss
                    if 'pattern_status' not in pattern:
                        pattern['pattern_status'] = 'active'
                        pattern['entry_price'] = pattern.get('breakout_price', price_data['close'].iloc[breakout_idx])
                    
                    # Skip already completed patterns
                    if pattern['pattern_status'] in ['target_reached', 'stop_loss_hit', 'expired']:
                        active_patterns.append(pattern)
                        continue
                    
                    # Check latest price against targets and stop loss
                    latest_price = price_data['close'].iloc[-1]
                    primary_target = pattern.get('targets', {}).get('primary')
                    stop_loss = pattern.get('stop_loss')
                    
                    # For bullish patterns
                    if pattern.get('direction') == 'bullish':
                        if primary_target and latest_price >= primary_target:
                            pattern['pattern_status'] = 'target_reached'
                            pattern['exit_price'] = latest_price
                            pattern['exit_date'] = price_data.index[-1].isoformat()
                            pattern['profit_pct'] = (latest_price - pattern['entry_price']) / pattern['entry_price']
                            self.pattern_history[symbol].append(pattern.copy())
                        elif stop_loss and latest_price <= stop_loss:
                            pattern['pattern_status'] = 'stop_loss_hit'
                            pattern['exit_price'] = latest_price
                            pattern['exit_date'] = price_data.index[-1].isoformat()
                            pattern['profit_pct'] = (latest_price - pattern['entry_price']) / pattern['entry_price']
                            self.pattern_history[symbol].append(pattern.copy())
                        else:
                            active_patterns.append(pattern)
                    
                    # For bearish patterns
                    elif pattern.get('direction') == 'bearish':
                        if primary_target and latest_price <= primary_target:
                            pattern['pattern_status'] = 'target_reached'
                            pattern['exit_price'] = latest_price
                            pattern['exit_date'] = price_data.index[-1].isoformat()
                            pattern['profit_pct'] = (pattern['entry_price'] - latest_price) / pattern['entry_price']
                            self.pattern_history[symbol].append(pattern.copy())
                        elif stop_loss and latest_price >= stop_loss:
                            pattern['pattern_status'] = 'stop_loss_hit'
                            pattern['exit_price'] = latest_price
                            pattern['exit_date'] = price_data.index[-1].isoformat()
                            pattern['profit_pct'] = (pattern['entry_price'] - latest_price) / pattern['entry_price']
                            self.pattern_history[symbol].append(pattern.copy())
                        else:
                            active_patterns.append(pattern)
                else:
                    # Pattern without breakout index is still forming
                    active_patterns.append(pattern)
            
            # Update the active patterns list
            self.detected_patterns[timeframe][symbol] = active_patterns
    
    def get_pattern_stats(self):
        """
        Get statistics on pattern performance.
        
        Returns:
            dict: Statistics on pattern performance
        """
        if not self.pattern_history:
            return {"message": "No pattern history available"}
        
        stats = {
            "total_patterns": 0,
            "success_rate": 0,
            "avg_profit_pct": 0,
            "by_pattern_type": {},
            "by_timeframe": {}
        }
        
        all_patterns = []
        for symbol, patterns in self.pattern_history.items():
            all_patterns.extend(patterns)
        
        stats["total_patterns"] = len(all_patterns)
        
        # Calculate overall success rate
        successful = [p for p in all_patterns if p.get('pattern_status') == 'target_reached']
        failed = [p for p in all_patterns if p.get('pattern_status') == 'stop_loss_hit']
        
        if successful or failed:
            success_rate = len(successful) / (len(successful) + len(failed)) if (len(successful) + len(failed)) > 0 else 0
            stats["success_rate"] = round(success_rate, 2)
        
        # Calculate average profit
        completed = [p for p in all_patterns if p.get('pattern_status') in ['target_reached', 'stop_loss_hit']]
        if completed:
            avg_profit = sum(p.get('profit_pct', 0) for p in completed) / len(completed)
            stats["avg_profit_pct"] = round(avg_profit, 4)
        
        # Break down by pattern type
        pattern_types = set(p.get('pattern') for p in all_patterns)
        for pattern_type in pattern_types:
            patterns_of_type = [p for p in all_patterns if p.get('pattern') == pattern_type]
            successful_of_type = [p for p in patterns_of_type if p.get('pattern_status') == 'target_reached']
            failed_of_type = [p for p in patterns_of_type if p.get('pattern_status') == 'stop_loss_hit']
            
            type_stats = {
                "count": len(patterns_of_type),
                "success_rate": round(len(successful_of_type) / (len(successful_of_type) + len(failed_of_type)), 2) 
                               if (len(successful_of_type) + len(failed_of_type)) > 0 else 0
            }
            
            completed_of_type = [p for p in patterns_of_type if p.get('pattern_status') in ['target_reached', 'stop_loss_hit']]
            if completed_of_type:
                type_stats["avg_profit_pct"] = round(sum(p.get('profit_pct', 0) for p in completed_of_type) / len(completed_of_type), 4)
            
            stats["by_pattern_type"][pattern_type] = type_stats
        
        # Break down by timeframe
        timeframes = set(p.get('timeframe') for p in all_patterns)
        for timeframe in timeframes:
            patterns_of_timeframe = [p for p in all_patterns if p.get('timeframe') == timeframe]
            successful_of_timeframe = [p for p in patterns_of_timeframe if p.get('pattern_status') == 'target_reached']
            failed_of_timeframe = [p for p in patterns_of_timeframe if p.get('pattern_status') == 'stop_loss_hit']
            
            timeframe_stats = {
                "count": len(patterns_of_timeframe),
                "success_rate": round(len(successful_of_timeframe) / (len(successful_of_timeframe) + len(failed_of_timeframe)), 2)
                               if (len(successful_of_timeframe) + len(failed_of_timeframe)) > 0 else 0
            }
            
            completed_of_timeframe = [p for p in patterns_of_timeframe if p.get('pattern_status') in ['target_reached', 'stop_loss_hit']]
            if completed_of_timeframe:
                timeframe_stats["avg_profit_pct"] = round(sum(p.get('profit_pct', 0) for p in completed_of_timeframe) / len(completed_of_timeframe), 4)
            
            stats["by_timeframe"][timeframe] = timeframe_stats
        
        return stats 