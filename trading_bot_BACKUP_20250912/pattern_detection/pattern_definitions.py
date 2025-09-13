import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

class ChartPattern(ABC):
    """Base class for all chart patterns."""
    
    def __init__(self, name, pattern_id, pattern_type, direction, reliability_score):
        self.name = name
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type  # 'continuation' or 'reversal'
        self.direction = direction  # 'bullish' or 'bearish'
        self.reliability_score = reliability_score
        self.timeframe_effectiveness = {}
        self.detection_parameters = {}
    
    @abstractmethod
    def detect(self, price_data, volume_data):
        """
        Detect the pattern in price data.
        
        Args:
            price_data (pd.DataFrame): Price data with OHLC
            volume_data (np.array): Volume data
            
        Returns:
            list: List of detected patterns with metadata
        """
        pass
    
    def calculate_target(self, pattern_data):
        """
        Calculate price targets based on pattern height.
        
        Args:
            pattern_data (dict): Pattern metadata
            
        Returns:
            dict: Target prices at different projection levels
        """
        pass
    
    def validate_pattern(self, pattern_data, market_condition=None):
        """
        Validate pattern against ideal and disqualifying conditions.
        
        Args:
            pattern_data (dict): Pattern metadata
            market_condition (str, optional): Current market condition
            
        Returns:
            tuple: (is_valid, confidence_score, validation_details)
        """
        pass

class DoubleBottom(ChartPattern):
    """Double Bottom pattern implementation."""
    
    def __init__(self):
        super().__init__(
            name="Double Bottom",
            pattern_id="DB001",
            pattern_type="reversal",
            direction="bullish",
            reliability_score=0.75
        )
        
        self.timeframe_effectiveness = {
            "intraday": 0.60,
            "daily": 0.80,
            "weekly": 0.85
        }
        
        self.detection_parameters = {
            "minimum_pattern_duration": 10,  # bars
            "maximum_pattern_duration": 40,  # bars
            "low_point_equality_threshold": 0.03,  # Maximum percentage difference between two lows
            "minimum_pattern_height": 0.04,  # As percentage of price
            "confirmation_volume_increase": 1.5,  # Multiple of average volume required on breakout
            "required_prior_trend_bars": 10  # Minimum bars in prior downtrend
        }
    
    def detect(self, price_data, volume_data):
        """Detect Double Bottom patterns in the price data."""
        patterns = []
        
        # Find local minima with 10-bar window
        lows = price_data['low'].values
        close = price_data['close'].values
        highs = price_data['high'].values
        
        # Find local minima
        minima_indices = find_peaks(-lows, distance=5)[0]
        
        # Filter for potential bottoms
        potential_bottoms = []
        for i, min_idx in enumerate(minima_indices[:-1]):
            for j, next_idx in enumerate(minima_indices[i+1:i+10]):  # Look only at nearby minima
                # Check if bottoms are roughly equal (within threshold)
                if (abs(lows[min_idx] - lows[next_idx]) / lows[min_idx] 
                    <= self.detection_parameters["low_point_equality_threshold"]):
                    
                    # Check minimum separation
                    if next_idx - min_idx >= self.detection_parameters["minimum_pattern_duration"]:
                        # Check maximum separation
                        if next_idx - min_idx <= self.detection_parameters["maximum_pattern_duration"]:
                            # Verify there's a peak between bottoms
                            between_indices = range(min_idx+1, next_idx)
                            if between_indices:
                                max_between = np.max(highs[between_indices])
                                peak_idx = min_idx + 1 + np.argmax(highs[between_indices])
                                if max_between > lows[min_idx] * 1.03:  # At least 3% above valleys
                                    potential_bottoms.append((min_idx, next_idx, peak_idx))
        
        # Check for confirmations and create patterns
        for first_idx, second_idx, neckline_idx in potential_bottoms:
            # Find the neckline (peak between bottoms)
            neckline_value = highs[neckline_idx]
            
            # Check for breakout after second bottom
            for i in range(second_idx + 1, min(len(close), second_idx + 20)):  # Look ahead maximum 20 bars
                if close[i] > neckline_value:
                    # Check volume confirmation on breakout
                    avg_volume = np.mean(volume_data[i-20:i])
                    breakout_volume = volume_data[i]
                    
                    if breakout_volume > avg_volume * self.detection_parameters["confirmation_volume_increase"]:
                        # Calculate pattern measurements
                        pattern_height = neckline_value - lows[second_idx]
                        pattern_height_pct = pattern_height / neckline_value
                        
                        # Check minimum pattern height
                        if pattern_height_pct >= self.detection_parameters["minimum_pattern_height"]:
                            # Calculate targets
                            primary_target = neckline_value + pattern_height
                            conservative_target = neckline_value + (pattern_height * 0.5)
                            aggressive_target = neckline_value + (pattern_height * 1.5)
                            
                            # Calculate stop loss
                            stop_loss = lows[second_idx] * 0.99  # 1% below second bottom
                            
                            patterns.append({
                                'pattern': 'double_bottom',
                                'pattern_id': self.pattern_id,
                                'direction': self.direction,
                                'first_bottom_idx': first_idx,
                                'second_bottom_idx': second_idx,
                                'neckline_idx': neckline_idx,
                                'breakout_idx': i,
                                'neckline_value': neckline_value,
                                'first_bottom_value': lows[first_idx],
                                'second_bottom_value': lows[second_idx],
                                'breakout_price': close[i],
                                'pattern_height': pattern_height,
                                'pattern_height_pct': pattern_height_pct,
                                'targets': {
                                    'conservative': conservative_target,
                                    'primary': primary_target,
                                    'aggressive': aggressive_target
                                },
                                'stop_loss': stop_loss,
                                'volume_confirmation': breakout_volume > avg_volume * self.detection_parameters["confirmation_volume_increase"],
                                'reliability_score': self.reliability_score
                            })
                            break
        
        return patterns
    
    def calculate_target(self, pattern_data):
        """Calculate price targets for Double Bottom pattern."""
        neckline = pattern_data['neckline_value']
        pattern_height = pattern_data['pattern_height']
        
        return {
            'conservative': neckline + (pattern_height * 0.5),
            'primary': neckline + pattern_height,
            'aggressive': neckline + (pattern_height * 1.5)
        }
    
    def validate_pattern(self, pattern_data, market_condition=None):
        """Validate Double Bottom pattern against ideal conditions."""
        confidence_score = self.reliability_score
        validation_details = []
        
        # Check if second bottom is significantly lower than first (disqualifying)
        if pattern_data['second_bottom_value'] < pattern_data['first_bottom_value'] * 0.95:
            validation_details.append("Second bottom significantly lower than first (>5%)")
            confidence_score *= 0.7
        
        # Check volume confirmation
        if not pattern_data['volume_confirmation']:
            validation_details.append("Lacking volume confirmation on breakout")
            confidence_score *= 0.8
        
        # Return the validation result
        is_valid = confidence_score >= 0.5  # Threshold for validity
        return (is_valid, confidence_score, validation_details)

class DoubleTop(ChartPattern):
    """Double Top pattern implementation."""
    
    def __init__(self):
        super().__init__(
            name="Double Top",
            pattern_id="DT001",
            pattern_type="reversal",
            direction="bearish",
            reliability_score=0.72
        )
        
        self.timeframe_effectiveness = {
            "intraday": 0.58,
            "daily": 0.75,
            "weekly": 0.80
        }
        
        self.detection_parameters = {
            "minimum_pattern_duration": 10,  # bars
            "maximum_pattern_duration": 40,  # bars
            "high_point_equality_threshold": 0.03,  # Maximum percentage difference between two highs
            "minimum_pattern_height": 0.04,  # As percentage of price
            "confirmation_volume_increase": 1.4,  # Multiple of average volume required on breakdown
            "required_prior_trend_bars": 10  # Minimum bars in prior uptrend
        }
    
    def detect(self, price_data, volume_data):
        """Detect Double Top patterns in the price data."""
        patterns = []
        
        # Find local maxima with 10-bar window
        highs = price_data['high'].values
        close = price_data['close'].values
        lows = price_data['low'].values
        
        # Find local maxima
        maxima_indices = find_peaks(highs, distance=5)[0]
        
        # Filter for potential tops
        potential_tops = []
        for i, max_idx in enumerate(maxima_indices[:-1]):
            for j, next_idx in enumerate(maxima_indices[i+1:i+10]):  # Look only at nearby maxima
                # Check if tops are roughly equal (within threshold)
                if (abs(highs[max_idx] - highs[next_idx]) / highs[max_idx] 
                    <= self.detection_parameters["high_point_equality_threshold"]):
                    
                    # Check minimum separation
                    if next_idx - max_idx >= self.detection_parameters["minimum_pattern_duration"]:
                        # Check maximum separation
                        if next_idx - max_idx <= self.detection_parameters["maximum_pattern_duration"]:
                            # Verify there's a valley between tops
                            between_indices = range(max_idx+1, next_idx)
                            if between_indices:
                                min_between = np.min(lows[between_indices])
                                valley_idx = max_idx + 1 + np.argmin(lows[between_indices])
                                if min_between < highs[max_idx] * 0.97:  # At least 3% below peaks
                                    potential_tops.append((max_idx, next_idx, valley_idx))
        
        # Check for confirmations and create patterns
        for first_idx, second_idx, neckline_idx in potential_tops:
            # Find the neckline (valley between tops)
            neckline_value = lows[neckline_idx]
            
            # Check for breakdown after second top
            for i in range(second_idx + 1, min(len(close), second_idx + 20)):  # Look ahead maximum 20 bars
                if close[i] < neckline_value:
                    # Check volume confirmation on breakdown
                    avg_volume = np.mean(volume_data[i-20:i])
                    breakdown_volume = volume_data[i]
                    
                    if breakdown_volume > avg_volume * self.detection_parameters["confirmation_volume_increase"]:
                        # Calculate pattern measurements
                        pattern_height = highs[second_idx] - neckline_value
                        pattern_height_pct = pattern_height / highs[second_idx]
                        
                        # Check minimum pattern height
                        if pattern_height_pct >= self.detection_parameters["minimum_pattern_height"]:
                            # Calculate targets
                            primary_target = neckline_value - pattern_height
                            conservative_target = neckline_value - (pattern_height * 0.5)
                            aggressive_target = neckline_value - (pattern_height * 1.5)
                            
                            # Calculate stop loss
                            stop_loss = highs[second_idx] * 1.01  # 1% above second top
                            
                            patterns.append({
                                'pattern': 'double_top',
                                'pattern_id': self.pattern_id,
                                'direction': self.direction,
                                'first_top_idx': first_idx,
                                'second_top_idx': second_idx,
                                'neckline_idx': neckline_idx,
                                'breakdown_idx': i,
                                'neckline_value': neckline_value,
                                'first_top_value': highs[first_idx],
                                'second_top_value': highs[second_idx],
                                'breakdown_price': close[i],
                                'pattern_height': pattern_height,
                                'pattern_height_pct': pattern_height_pct,
                                'targets': {
                                    'conservative': conservative_target,
                                    'primary': primary_target,
                                    'aggressive': aggressive_target
                                },
                                'stop_loss': stop_loss,
                                'volume_confirmation': breakdown_volume > avg_volume * self.detection_parameters["confirmation_volume_increase"],
                                'reliability_score': self.reliability_score
                            })
                            break
        
        return patterns

class BullFlag(ChartPattern):
    """Bull Flag pattern implementation."""
    
    def __init__(self):
        super().__init__(
            name="Bull Flag",
            pattern_id="BF001",
            pattern_type="continuation",
            direction="bullish",
            reliability_score=0.82
        )
        
        self.timeframe_effectiveness = {
            "intraday": 0.80,
            "daily": 0.83,
            "weekly": 0.75
        }
        
        self.detection_parameters = {
            "minimum_flagpole_height": 0.08,  # Minimum height as percentage of price
            "maximum_flagpole_duration": 10,  # Flagpole shouldn't take too long to form
            "minimum_flagpole_duration": 3,   # Flagpole shouldn't be too short
            "minimum_flag_duration": 5,      # Flag consolidation period
            "maximum_flag_duration": 20,     # Flag shouldn't take too long to form
            "maximum_flag_height": 0.5,      # Flag height as percentage of flagpole height
            "maximum_flag_slope": -0.01,     # Flag should have downward slope
            "minimum_volume_ratio_flagpole": 1.5  # Flagpole volume vs. pre-flagpole average
        }
    
    def detect(self, price_data, volume_data):
        """Detect Bull Flag patterns in the price data."""
        patterns = []
        close = price_data['close'].values
        high = price_data['high'].values
        low = price_data['low'].values
        
        # Scan for potential flagpoles (sharp rallies)
        for i in range(10, len(close) - 25):  # Need room for both pole and flag
            # Check for flagpole (strong upward move)
            start_idx = i - 10
            potential_pole_end = i
            
            # Calculate pole height and duration
            pole_start_price = low[start_idx]
            pole_end_price = high[potential_pole_end]
            pole_height = (pole_end_price - pole_start_price) / pole_start_price
            pole_duration = potential_pole_end - start_idx
            
            # Check volume during pole
            pole_volume_avg = np.mean(volume_data[start_idx:potential_pole_end+1])
            pre_pole_volume_avg = np.mean(volume_data[max(0, start_idx-10):start_idx])
            pole_volume_ratio = pole_volume_avg / pre_pole_volume_avg if pre_pole_volume_avg > 0 else 0
            
            # If we have a valid flagpole, look for flag
            if (pole_height > self.detection_parameters["minimum_flagpole_height"] and 
                pole_duration >= self.detection_parameters["minimum_flagpole_duration"] and 
                pole_duration <= self.detection_parameters["maximum_flagpole_duration"] and 
                pole_volume_ratio > self.detection_parameters["minimum_volume_ratio_flagpole"]):
                
                # Look for flag (consolidation) after pole
                for flag_end in range(potential_pole_end + 5, 
                                     min(potential_pole_end + self.detection_parameters["maximum_flag_duration"] + 1, 
                                         len(close))):
                    flag_duration = flag_end - potential_pole_end
                    
                    # Check if flag duration is reasonable
                    if (flag_duration >= self.detection_parameters["minimum_flag_duration"] and 
                        flag_duration <= self.detection_parameters["maximum_flag_duration"]):
                        
                        # Get flag high and low prices
                        flag_highs = high[potential_pole_end:flag_end+1]
                        flag_lows = low[potential_pole_end:flag_end+1]
                        
                        # Fit downward sloping channel to flag
                        X = np.array(range(len(flag_highs))).reshape(-1, 1)
                        
                        # Fit upper line to highs
                        upper_regressor = LinearRegression().fit(X, flag_highs)
                        upper_slope = upper_regressor.coef_[0]
                        
                        # Fit lower line to lows
                        lower_regressor = LinearRegression().fit(X, flag_lows)
                        lower_slope = lower_regressor.coef_[0]
                        
                        # Check for downward slope
                        if upper_slope < 0 and lower_slope < 0:
                            # Calculate flag height
                            flag_height = max(flag_highs) - min(flag_lows)
                            flag_to_pole_ratio = flag_height / (pole_end_price - pole_start_price)
                            
                            # Check flag height relative to pole
                            if flag_to_pole_ratio < self.detection_parameters["maximum_flag_height"]:
                                # Check flag volume (should decrease)
                                flag_volume_avg = np.mean(volume_data[potential_pole_end:flag_end+1])
                                
                                if flag_volume_avg < pole_volume_avg:
                                    # Look for breakout
                                    for j in range(flag_end, min(flag_end + 15, len(close))):
                                        # Calculate expected upper trendline value at this point
                                        breakout_x = j - potential_pole_end
                                        expected_upper = upper_regressor.intercept_ + upper_regressor.coef_[0] * breakout_x
                                        
                                        if close[j] > expected_upper:
                                            # Check volume on breakout
                                            breakout_volume = volume_data[j]
                                            recent_avg_volume = np.mean(volume_data[max(0, j-20):j])
                                            
                                            if breakout_volume > recent_avg_volume * 1.5:
                                                # Calculate target and stop loss
                                                target = close[j] + (pole_end_price - pole_start_price)
                                                stop_loss = min(flag_lows) * 0.99
                                                
                                                patterns.append({
                                                    'pattern': 'bull_flag',
                                                    'pattern_id': self.pattern_id,
                                                    'direction': self.direction,
                                                    'pole_start_idx': start_idx,
                                                    'pole_end_idx': potential_pole_end,
                                                    'flag_end_idx': flag_end,
                                                    'breakout_idx': j,
                                                    'pole_start_price': pole_start_price,
                                                    'pole_end_price': pole_end_price,
                                                    'pole_height': pole_height,
                                                    'flag_height': flag_height,
                                                    'flag_to_pole_ratio': flag_to_pole_ratio,
                                                    'breakout_price': close[j],
                                                    'targets': {
                                                        'conservative': close[j] + (pole_end_price - pole_start_price) * 0.7,
                                                        'primary': close[j] + (pole_end_price - pole_start_price),
                                                        'aggressive': close[j] + (pole_end_price - pole_start_price) * 1.5
                                                    },
                                                    'stop_loss': stop_loss,
                                                    'volume_confirmation': breakout_volume > recent_avg_volume * 1.5,
                                                    'reliability_score': self.reliability_score
                                                })
                                                break
        
        return patterns

class BearFlag(ChartPattern):
    """Bear Flag pattern implementation."""
    
    def __init__(self):
        super().__init__(
            name="Bear Flag",
            pattern_id="BRF001",
            pattern_type="continuation",
            direction="bearish",
            reliability_score=0.80
        )
        
        self.timeframe_effectiveness = {
            "intraday": 0.78,
            "daily": 0.82,
            "weekly": 0.74
        }
        
        self.detection_parameters = {
            "minimum_flagpole_height": 0.08,  # Minimum height as percentage of price
            "maximum_flagpole_duration": 10,  # Flagpole shouldn't take too long to form
            "minimum_flagpole_duration": 3,   # Flagpole shouldn't be too short
            "minimum_flag_duration": 5,      # Flag consolidation period
            "maximum_flag_duration": 20,     # Flag shouldn't take too long to form
            "maximum_flag_height": 0.5,      # Flag height as percentage of flagpole height
            "minimum_flag_slope": 0.01,      # Flag should have upward slope
            "minimum_volume_ratio_flagpole": 1.5  # Flagpole volume vs. pre-flagpole average
        }
    
    def detect(self, price_data, volume_data):
        """Detect Bear Flag patterns in the price data."""
        patterns = []
        close = price_data['close'].values
        high = price_data['high'].values
        low = price_data['low'].values
        
        # Scan for potential flagpoles (sharp declines)
        for i in range(10, len(close) - 25):  # Need room for both pole and flag
            # Check for flagpole (strong downward move)
            start_idx = i - 10
            potential_pole_end = i
            
            # Calculate pole height and duration
            pole_start_price = high[start_idx]
            pole_end_price = low[potential_pole_end]
            pole_height = (pole_start_price - pole_end_price) / pole_start_price  # Percentage decline
            pole_duration = potential_pole_end - start_idx
            
            # Check volume during pole
            pole_volume_avg = np.mean(volume_data[start_idx:potential_pole_end+1])
            pre_pole_volume_avg = np.mean(volume_data[max(0, start_idx-10):start_idx])
            pole_volume_ratio = pole_volume_avg / pre_pole_volume_avg if pre_pole_volume_avg > 0 else 0
            
            # If we have a valid flagpole, look for flag
            if (pole_height > self.detection_parameters["minimum_flagpole_height"] and 
                pole_duration >= self.detection_parameters["minimum_flagpole_duration"] and 
                pole_duration <= self.detection_parameters["maximum_flagpole_duration"] and 
                pole_volume_ratio > self.detection_parameters["minimum_volume_ratio_flagpole"]):
                
                # Look for flag (consolidation) after pole
                for flag_end in range(potential_pole_end + 5, 
                                     min(potential_pole_end + self.detection_parameters["maximum_flag_duration"] + 1, 
                                         len(close))):
                    flag_duration = flag_end - potential_pole_end
                    
                    # Check if flag duration is reasonable
                    if (flag_duration >= self.detection_parameters["minimum_flag_duration"] and 
                        flag_duration <= self.detection_parameters["maximum_flag_duration"]):
                        
                        # Get flag high and low prices
                        flag_highs = high[potential_pole_end:flag_end+1]
                        flag_lows = low[potential_pole_end:flag_end+1]
                        
                        # Fit upward sloping channel to flag
                        X = np.array(range(len(flag_highs))).reshape(-1, 1)
                        
                        # Fit upper line to highs
                        upper_regressor = LinearRegression().fit(X, flag_highs)
                        upper_slope = upper_regressor.coef_[0]
                        
                        # Fit lower line to lows
                        lower_regressor = LinearRegression().fit(X, flag_lows)
                        lower_slope = lower_regressor.coef_[0]
                        
                        # Check for upward slope
                        if upper_slope > 0 and lower_slope > 0:
                            # Calculate flag height
                            flag_height = max(flag_highs) - min(flag_lows)
                            flag_to_pole_ratio = flag_height / (pole_start_price - pole_end_price)
                            
                            # Check flag height relative to pole
                            if flag_to_pole_ratio < self.detection_parameters["maximum_flag_height"]:
                                # Check flag volume (should decrease)
                                flag_volume_avg = np.mean(volume_data[potential_pole_end:flag_end+1])
                                
                                if flag_volume_avg < pole_volume_avg:
                                    # Look for breakdown
                                    for j in range(flag_end, min(flag_end + 15, len(close))):
                                        # Calculate expected lower trendline value at this point
                                        breakdown_x = j - potential_pole_end
                                        expected_lower = lower_regressor.intercept_ + lower_regressor.coef_[0] * breakdown_x
                                        
                                        if close[j] < expected_lower:
                                            # Check volume on breakdown
                                            breakdown_volume = volume_data[j]
                                            recent_avg_volume = np.mean(volume_data[max(0, j-20):j])
                                            
                                            if breakdown_volume > recent_avg_volume * 1.5:
                                                # Calculate target and stop loss
                                                target = close[j] - (pole_start_price - pole_end_price)
                                                stop_loss = max(flag_highs) * 1.01
                                                
                                                patterns.append({
                                                    'pattern': 'bear_flag',
                                                    'pattern_id': self.pattern_id,
                                                    'direction': self.direction,
                                                    'pole_start_idx': start_idx,
                                                    'pole_end_idx': potential_pole_end,
                                                    'flag_end_idx': flag_end,
                                                    'breakdown_idx': j,
                                                    'pole_start_price': pole_start_price,
                                                    'pole_end_price': pole_end_price,
                                                    'pole_height': pole_height,
                                                    'flag_height': flag_height,
                                                    'flag_to_pole_ratio': flag_to_pole_ratio,
                                                    'breakdown_price': close[j],
                                                    'targets': {
                                                        'conservative': close[j] - (pole_start_price - pole_end_price) * 0.7,
                                                        'primary': close[j] - (pole_start_price - pole_end_price),
                                                        'aggressive': close[j] - (pole_start_price - pole_end_price) * 1.5
                                                    },
                                                    'stop_loss': stop_loss,
                                                    'volume_confirmation': breakdown_volume > recent_avg_volume * 1.5,
                                                    'reliability_score': self.reliability_score
                                                })
                                                break
        
        return patterns

class HeadAndShoulders(ChartPattern):
    """Head and Shoulders pattern implementation."""
    
    def __init__(self):
        super().__init__(
            name="Head and Shoulders",
            pattern_id="HS001",
            pattern_type="reversal",
            direction="bearish",
            reliability_score=0.80
        )
        
        self.timeframe_effectiveness = {
            "intraday": 0.65,
            "daily": 0.82,
            "weekly": 0.84
        }
        
        self.detection_parameters = {
            "minimum_pattern_duration": 20,
            "maximum_pattern_duration": 60,
            "minimum_head_prominence": 0.03,  # Head must be at least 3% higher than shoulders
            "maximum_shoulder_height_difference": 0.05,  # Left and right shoulders should be within 5% height
            "minimum_volume_profile_match": 0.70,  # Volume should follow classic volume pattern (score 0-1)
            "maximum_neckline_slope": 0.05,  # Neckline shouldn't slope too steeply (up or down)
            "minimum_pattern_height": 0.05  # Total height from neckline to head as percentage of price
        }
    
    def detect(self, price_data, volume_data):
        """Detect Head and Shoulders patterns in the price data."""
        patterns = []
        high = price_data['high'].values
        low = price_data['low'].values
        close = price_data['close'].values
        
        # Find potential patterns
        for i in range(30, len(close) - 20):
            # Look for sequences of peaks and troughs
            window_start = max(0, i - 30)
            window = high[window_start:i+20]
            
            # Detect local maxima (peaks)
            peaks = find_peaks(window, distance=5)[0]
            
            # Need at least 3 peaks for head and shoulders
            if len(peaks) >= 3:
                for j in range(len(peaks) - 2):
                    # Check if the middle peak is higher than surrounding peaks
                    left_peak_idx = peaks[j]
                    head_idx = peaks[j+1]
                    right_peak_idx = peaks[j+2]
                    
                    left_peak_value = window[left_peak_idx]
                    head_value = window[head_idx]
                    right_peak_value = window[right_peak_idx]
                    
                    # Convert local indices to window indices
                    abs_left_peak_idx = window_start + left_peak_idx
                    abs_head_idx = window_start + head_idx
                    abs_right_peak_idx = window_start + right_peak_idx
                    
                    # Pattern should have sufficient duration
                    pattern_duration = abs_right_peak_idx - abs_left_peak_idx
                    if (pattern_duration < self.detection_parameters["minimum_pattern_duration"] or
                        pattern_duration > self.detection_parameters["maximum_pattern_duration"]):
                        continue
                    
                    # Head must be higher than shoulders
                    if (head_value > left_peak_value and 
                        head_value > right_peak_value):
                        
                        # Check shoulder height similarity
                        shoulder_diff = abs(left_peak_value - right_peak_value) / left_peak_value
                        if shoulder_diff > self.detection_parameters["maximum_shoulder_height_difference"]:
                            continue
                        
                        # Head must be at least X% higher than shoulders
                        head_prominence = (head_value - max(left_peak_value, right_peak_value)) / max(left_peak_value, right_peak_value)
                        if head_prominence < self.detection_parameters["minimum_head_prominence"]:
                            continue
                        
                        # Find troughs between peaks (for neckline)
                        left_trough_segment = low[abs_left_peak_idx:abs_head_idx]
                        right_trough_segment = low[abs_head_idx:abs_right_peak_idx]
                        
                        if len(left_trough_segment) == 0 or len(right_trough_segment) == 0:
                            continue
                            
                        left_trough_local_idx = np.argmin(left_trough_segment)
                        right_trough_local_idx = np.argmin(right_trough_segment)
                        
                        abs_left_trough_idx = abs_left_peak_idx + left_trough_local_idx
                        abs_right_trough_idx = abs_head_idx + right_trough_local_idx
                        
                        left_trough_value = low[abs_left_trough_idx]
                        right_trough_value = low[abs_right_trough_idx]
                        
                        # Calculate neckline slope
                        neckline_slope = (right_trough_value - left_trough_value) / (abs_right_trough_idx - abs_left_trough_idx)
                        neckline_slope_pct = neckline_slope / left_trough_value
                        
                        # Check if neckline slope is acceptable
                        if abs(neckline_slope_pct) > self.detection_parameters["maximum_neckline_slope"]:
                            continue
                            
                        # Calculate pattern height
                        neckline_at_head = left_trough_value + neckline_slope * (abs_head_idx - abs_left_trough_idx)
                        pattern_height = head_value - neckline_at_head
                        pattern_height_pct = pattern_height / head_value
                        
                        # Check minimum pattern height
                        if pattern_height_pct < self.detection_parameters["minimum_pattern_height"]:
                            continue
                            
                        # Check volume pattern (should decrease from left to right)
                        left_shoulder_volume = np.mean(volume_data[max(0, abs_left_peak_idx-2):abs_left_peak_idx+3])
                        head_volume = np.mean(volume_data[max(0, abs_head_idx-2):abs_head_idx+3])
                        right_shoulder_volume = np.mean(volume_data[max(0, abs_right_peak_idx-2):min(len(volume_data), abs_right_peak_idx+3)])
                        
                        # Volume should gradually decrease
                        if not (head_volume <= left_shoulder_volume or right_shoulder_volume <= head_volume):
                            continue
                            
                        # Look for breakdown
                        breakdown_idx = None
                        for k in range(abs_right_peak_idx + 1, min(len(close), abs_right_peak_idx + 15)):
                            # Calculate neckline value at this bar
                            neckline_value = left_trough_value + neckline_slope * (k - abs_left_trough_idx)
                            
                            # Check for breakdown with volume
                            if (close[k] < neckline_value and
                                volume_data[k] > np.mean(volume_data[max(0, k-20):k]) * 1.5):
                                
                                breakdown_idx = k
                                breakdown_price = close[k]
                                
                                # Calculate targets and stop loss
                                primary_target = breakdown_price - pattern_height
                                conservative_target = breakdown_price - (pattern_height * 0.5)
                                aggressive_target = breakdown_price - (pattern_height * 2)
                                stop_loss = high[abs_right_peak_idx] * 1.01  # 1% above right shoulder
                                
                                patterns.append({
                                    'pattern': 'head_and_shoulders',
                                    'pattern_id': self.pattern_id,
                                    'direction': self.direction,
                                    'left_shoulder_idx': abs_left_peak_idx,
                                    'head_idx': abs_head_idx,
                                    'right_shoulder_idx': abs_right_peak_idx,
                                    'left_trough_idx': abs_left_trough_idx,
                                    'right_trough_idx': abs_right_trough_idx,
                                    'breakdown_idx': breakdown_idx,
                                    'left_shoulder_value': left_peak_value,
                                    'head_value': head_value,
                                    'right_shoulder_value': right_peak_value,
                                    'neckline_left_value': left_trough_value,
                                    'neckline_right_value': right_trough_value,
                                    'neckline_slope': neckline_slope,
                                    'pattern_height': pattern_height,
                                    'pattern_height_pct': pattern_height_pct,
                                    'breakdown_price': breakdown_price,
                                    'targets': {
                                        'conservative': conservative_target,
                                        'primary': primary_target,
                                        'aggressive': aggressive_target
                                    },
                                    'stop_loss': stop_loss,
                                    'reliability_score': self.reliability_score
                                })
                                break
        
        return patterns

# More pattern classes would be defined here following the same structure
# (CupAndHandle, AscendingTriangle, DescendingTriangle, InverseHeadAndShoulders, etc.)

def get_all_patterns():
    """Return all chart pattern detection classes."""
    return [
        DoubleBottom(),
        DoubleTop(),
        BullFlag(),
        BearFlag(),
        HeadAndShoulders()
        # Add other pattern classes here as they are implemented
    ] 