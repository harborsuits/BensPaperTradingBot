import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import our module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading_bot.backtesting.advanced_regime_detector import (
    AdvancedRegimeDetector,
    VolatilityRegime,
    CorrelationRegime,
    SectorRotationPhase
)

class TestAdvancedRegimeDetector(unittest.TestCase):
    """Test cases for the AdvancedRegimeDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Generate synthetic market data
        self.market_data = self._generate_test_data()
        
        # Create an instance of the detector
        self.detector = AdvancedRegimeDetector(
            lookback_days=60,  # Shorter window for testing
            timeframes=['daily', 'weekly'],
            volatility_windows={'daily': 10, 'weekly': 4},
            trend_windows={'daily': 20, 'weekly': 8},
            correlation_window=15,
            num_regimes=4,
            regime_persistence=3
        )
        
        # Load the data into the detector
        self.detector.load_market_data_multi_timeframe(
            price_data_by_timeframe=self.market_data,
            symbol_col='symbol',
            date_col='date',
            price_col='close',
            volume_col='volume',
            benchmark_symbol='SPY'
        )
    
    def _generate_test_data(self):
        """Generate synthetic market data for testing"""
        # Create date ranges
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=90)  # 90 days of history
        
        daily_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        weekly_dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
        
        # List of symbols to include
        symbols = ['SPY', 'QQQ', 'XLK', 'XLF', 'XLV', 'XLE', 'GLD']
        
        # Initialize data containers
        daily_data = []
        weekly_data = []
        
        # Generate price data for SPY (benchmark)
        np.random.seed(42)  # For reproducibility
        spy_daily_returns = np.random.normal(0.0005, 0.01, size=len(daily_dates))
        spy_daily_prices = 100 * np.cumprod(1 + spy_daily_returns)
        
        # Correlation coefficients for the test
        correlations = {
            'QQQ': 0.9,
            'XLK': 0.85,
            'XLF': 0.7,
            'XLV': 0.5,
            'XLE': 0.3,
            'GLD': -0.2
        }
        
        # Generate data for each symbol
        for symbol in symbols:
            # For SPY, use the base series
            if symbol == 'SPY':
                price_series = spy_daily_prices
            else:
                # Generate correlated returns
                correlation = correlations.get(symbol, 0.5)
                independent_returns = np.random.normal(0.0003, 0.012, size=len(daily_dates))
                correlated_returns = (correlation * spy_daily_returns + 
                                     np.sqrt(1 - correlation**2) * independent_returns)
                
                # Convert returns to prices
                price_series = 100 * np.cumprod(1 + correlated_returns)
            
            # Create daily data entries
            for i, date in enumerate(daily_dates):
                daily_data.append({
                    'date': date,
                    'symbol': symbol,
                    'close': price_series[i],
                    'volume': np.random.randint(100000, 10000000)
                })
            
            # Create weekly data (use Friday's values)
            weekly_indices = [i for i, date in enumerate(daily_dates) if date.dayofweek == 4 and date in weekly_dates]
            for i in weekly_indices:
                if i < len(price_series):
                    weekly_data.append({
                        'date': daily_dates[i],
                        'symbol': symbol,
                        'close': price_series[i],
                        'volume': np.random.randint(500000, 50000000)
                    })
        
        # Convert to DataFrames
        daily_df = pd.DataFrame(daily_data)
        weekly_df = pd.DataFrame(weekly_data)
        
        # Create a dictionary mapping timeframes to price DataFrames
        data_by_timeframe = {
            'daily': daily_df,
            'weekly': weekly_df
        }
        
        return data_by_timeframe
    
    def test_multi_timeframe_features(self):
        """Test computing features for multiple timeframes"""
        # Compute features
        features = self.detector.compute_features_multi_timeframe()
        
        # Check that features were computed for both timeframes
        self.assertIn('daily', features)
        self.assertIn('weekly', features)
        
        # Check that feature DataFrames contain expected columns
        expected_cols = ['volatility', 'trend_strength', 'avg_correlation', 
                         'return_dispersion', 'mean_reversion', 'hurst_exponent', 'vix_proxy']
        
        for timeframe, feature_df in features.items():
            for col in expected_cols:
                self.assertIn(col, feature_df.columns)
    
    def test_multi_timeframe_regime_detection(self):
        """Test detecting regimes for multiple timeframes"""
        # Compute features first
        self.detector.compute_features_multi_timeframe()
        
        # Detect regimes
        regimes = self.detector.detect_regimes_multi_timeframe()
        
        # Check that regimes were detected for both timeframes
        self.assertIn('daily', regimes)
        self.assertIn('weekly', regimes)
        
        # Check that the regimes are Series with some values
        for timeframe, regime_series in regimes.items():
            self.assertIsInstance(regime_series, pd.Series)
            self.assertGreater(len(regime_series), 0)
    
    def test_trend_conflicts(self):
        """Test detecting trend conflicts across timeframes"""
        # Run the prerequisite steps
        self.detector.compute_features_multi_timeframe()
        self.detector.detect_regimes_multi_timeframe()
        
        # Detect trend conflicts
        conflicts = self.detector.detect_trend_conflicts()
        
        # Check that the result is a DataFrame
        self.assertIsInstance(conflicts, pd.DataFrame)
        
        # Check that it has the expected columns
        expected_cols = ['primary_regime', 'primary_regime_label', 
                        'weekly_regime', 'weekly_regime_label', 'conflict_status']
        
        for col in expected_cols:
            self.assertIn(col, conflicts.columns)
        
        # Check that the conflict_status column contains valid values
        valid_statuses = ['aligned_bullish', 'aligned_bearish', 'strong_conflict', 'indeterminate']
        self.assertTrue(all(status in valid_statuses for status in conflicts['conflict_status'].unique()))
    
    def test_volatility_regimes(self):
        """Test classifying volatility regimes"""
        # Run the prerequisite steps
        self.detector.compute_features_multi_timeframe()
        
        # Classify volatility regimes
        vol_regimes = self.detector.classify_volatility_regimes()
        
        # Check that the result is a Series
        self.assertIsInstance(vol_regimes, pd.Series)
        
        # Check that it contains valid regime values
        valid_regimes = [VolatilityRegime.LOW, VolatilityRegime.NORMAL, 
                         VolatilityRegime.HIGH, VolatilityRegime.EXTREME]
        
        self.assertTrue(all(regime in valid_regimes for regime in vol_regimes.unique()))
    
    def test_correlation_regimes(self):
        """Test detecting correlation regimes"""
        # Run the prerequisite steps
        self.detector.compute_features_multi_timeframe()
        
        # Detect correlation regimes
        corr_regimes = self.detector.detect_correlation_regimes()
        
        # Check that the result is a DataFrame
        self.assertIsInstance(corr_regimes, pd.DataFrame)
        
        # Check that it contains at least some asset columns (not _value columns)
        asset_cols = [col for col in corr_regimes.columns if '_value' not in col]
        self.assertGreater(len(asset_cols), 0)
        
        # Check that the value columns exist
        for asset in asset_cols:
            self.assertIn(f"{asset}_value", corr_regimes.columns)
    
    def test_sector_rotation(self):
        """Test analyzing sector rotation"""
        # Run the prerequisite steps
        self.detector.compute_features_multi_timeframe()
        
        # Analyze sector rotation
        rotation = self.detector.analyze_sector_rotation()
        
        # Check that the result is a DataFrame
        self.assertIsInstance(rotation, pd.DataFrame)
        
        # Check that it contains the expected columns
        expected_cols = ['top_sectors', 'bottom_sectors', 'rotation_phase', 'sector_dispersion']
        for date in rotation.index:
            row = rotation.loc[date]
            for col in expected_cols:
                if pd.notna(row[col]):
                    if col == 'rotation_phase':
                        self.assertIn(row[col], [
                            SectorRotationPhase.EARLY_EXPANSION,
                            SectorRotationPhase.LATE_EXPANSION,
                            SectorRotationPhase.EARLY_CONTRACTION,
                            SectorRotationPhase.LATE_CONTRACTION,
                            SectorRotationPhase.RECOVERY,
                            "indeterminate"
                        ])
    
    def test_full_analysis(self):
        """Test running the full analysis pipeline"""
        # Run full analysis
        analysis = self.detector.run_full_analysis()
        
        # Check that the result is a dictionary
        self.assertIsInstance(analysis, dict)
        
        # Check that it contains the expected keys
        expected_keys = ['timestamp', 'timeframes_analyzed', 'primary_timeframe']
        
        for key in expected_keys:
            self.assertIn(key, analysis)
        
        # Check some of the expected analysis components
        if 'trend_conflicts' in analysis:
            self.assertIn('status', analysis['trend_conflicts'])
        
        if 'volatility_regime' in analysis:
            self.assertIn('current', analysis['volatility_regime'])
        
        if 'correlation_regimes' in analysis:
            self.assertIn('current', analysis['correlation_regimes'])
        
        if 'sector_rotation' in analysis:
            self.assertIn('current_phase', analysis['sector_rotation'])
        
        # Check that actionable insights are provided
        self.assertIn('actionable_insights', analysis)
        if analysis['actionable_insights']:
            insight = analysis['actionable_insights'][0]
            self.assertIn('type', insight)
            self.assertIn('description', insight)
            self.assertIn('recommendation', insight)

if __name__ == '__main__':
    unittest.main() 