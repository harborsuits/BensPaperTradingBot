import logging
import json
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

class MarketContextProvider:
    """
    Provides market context information by analyzing current market conditions,
    news sentiment, technical indicators, and macroeconomic data.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the Market Context Provider.
        
        Args:
            config_path (str, optional): Path to the configuration file.
        """
        self.logger = logging.getLogger(__name__)
        self._load_config(config_path)
        self.last_context_update = None
        self.current_context = {}
        self.context_history = []
        self.max_history_length = 100
        
    def _load_config(self, config_path):
        """Load configuration from file or use defaults."""
        default_config = {
            'update_interval_minutes': 60,
            'market_regime_lookback_days': 90,
            'volatility_calc_window': 21,
            'trend_calc_window': 14,
            'sentiment_weight': 0.3,
            'technical_weight': 0.4,
            'fundamental_weight': 0.3,
            'volatility_thresholds': {
                'low': 10.0,
                'high': 20.0
            },
            'trend_strength_thresholds': {
                'weak': 0.2,
                'strong': 0.6
            },
            'data_sources': {
                'price_data': 'alpha_vantage',
                'news_data': 'news_api',
                'economic_data': 'fred'
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                self.config = {**default_config, **user_config}
                self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
                self.config = default_config
        else:
            self.logger.warning(f"Config path not provided or not found. Using defaults.")
            self.config = default_config
            
    def get_current_market_context(self, force_update=False) -> Dict[str, Any]:
        """
        Get the current market context.
        
        Args:
            force_update (bool): Force a new update regardless of time interval.
            
        Returns:
            dict: Current market context with various metrics and indicators.
        """
        current_time = datetime.now()
        update_needed = (
            force_update or 
            self.last_context_update is None or
            (current_time - self.last_context_update).total_seconds() > 
            self.config['update_interval_minutes'] * 60
        )
        
        if update_needed:
            self.logger.info("Updating market context")
            try:
                new_context = self._generate_market_context()
                self.current_context = new_context
                self.last_context_update = current_time
                
                # Add to history
                self.context_history.append({
                    'timestamp': current_time.isoformat(),
                    'context': new_context
                })
                
                # Trim history if needed
                if len(self.context_history) > self.max_history_length:
                    self.context_history = self.context_history[-self.max_history_length:]
                    
            except Exception as e:
                self.logger.error(f"Error updating market context: {e}")
                if not self.current_context:
                    # Generate a neutral context if there's an error and no existing context
                    self.current_context = self._generate_neutral_context()
        
        return self.current_context
    
    def _generate_market_context(self) -> Dict[str, Any]:
        """
        Generate a comprehensive market context by analyzing various data sources.
        
        Returns:
            dict: Market context with multiple layers of analysis.
        """
        # Get market data
        try:
            market_data = self._get_market_data()
            technical_indicators = self._analyze_technical_indicators(market_data)
            sentiment_analysis = self._analyze_market_sentiment()
            macro_analysis = self._analyze_macroeconomic_data()
            volatility_analysis = self._analyze_market_volatility(market_data)
            trend_analysis = self._analyze_market_trends(market_data)
            correlation_analysis = self._analyze_cross_asset_correlations(market_data)
            
            # Determine overall market regime
            market_regime = self._determine_market_regime(
                technical_indicators, 
                sentiment_analysis, 
                volatility_analysis,
                trend_analysis
            )
            
            # Compile the full context
            context = {
                'market_regime': market_regime,
                'technical_indicators': technical_indicators,
                'sentiment': sentiment_analysis,
                'macroeconomic': macro_analysis,
                'market_volatility': volatility_analysis,
                'trend_strength': trend_analysis,
                'correlations': correlation_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error generating market context: {e}")
            return self._generate_neutral_context()
    
    def _get_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data from various sources.
        
        In a real implementation, this would retrieve actual market data
        from APIs or databases.
        
        Returns:
            dict: Market data organized by asset class and instrument.
        """
        # This is a mock implementation. In a real system, this would fetch
        # actual market data from APIs or databases.
        
        # Create sample data for demonstration
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        # Generate sample price data for different asset classes
        market_data = {
            'equities': {
                'SPY': self._generate_sample_price_data(dates, 400, 450),
                'QQQ': self._generate_sample_price_data(dates, 300, 350),
                'IWM': self._generate_sample_price_data(dates, 180, 220)
            },
            'fixed_income': {
                'TLT': self._generate_sample_price_data(dates, 120, 150),
                'IEF': self._generate_sample_price_data(dates, 100, 110)
            },
            'commodities': {
                'GLD': self._generate_sample_price_data(dates, 150, 180),
                'USO': self._generate_sample_price_data(dates, 70, 90)
            },
            'forex': {
                'EURUSD': self._generate_sample_price_data(dates, 1.05, 1.15),
                'USDJPY': self._generate_sample_price_data(dates, 100, 120)
            }
        }
        
        return market_data
    
    def _generate_sample_price_data(self, dates, min_price, max_price):
        """Generate sample OHLCV data for testing."""
        n = len(dates)
        
        # Generate a random walk with drift
        np.random.seed(int(min_price * 100))  # Ensure deterministic but different for each asset
        returns = np.random.normal(0.0002, 0.015, n)
        prices = min_price * np.exp(np.cumsum(returns))
        
        # Scale to the desired range
        scale_factor = (max_price - min_price) / (max(prices) - min(prices)) if max(prices) > min(prices) else 1
        prices = min_price + (prices - min(prices)) * scale_factor
        
        # Create OHLCV data
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, 0.005, n))
        df.loc[df.index[0], 'open'] = df.loc[df.index[0], 'close'] * 0.995
        
        # High and low based on open and close
        daily_range = abs(df['close'] - df['open']) * (1 + abs(np.random.normal(0, 0.5, n)))
        df['high'] = df[['open', 'close']].max(axis=1) + daily_range / 2
        df['low'] = df[['open', 'close']].min(axis=1) - daily_range / 2
        
        # Volume
        df['volume'] = np.random.lognormal(15, 1, n) * (1 + returns * 5)**2
        
        return df
    
    def _analyze_technical_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate various technical indicators from market data.
        
        Args:
            market_data (dict): Market data for various assets.
            
        Returns:
            dict: Technical indicators and their interpretations.
        """
        # This is a simplified implementation that would be expanded
        # in a production environment with more sophisticated indicators
        
        indicators = {
            'moving_averages': {},
            'oscillators': {},
            'volatility': {},
            'breadth': {}
        }
        
        # Analyze equity indices as proxies for the overall market
        for index_name, data in market_data.get('equities', {}).items():
            if not isinstance(data, pd.DataFrame) or data.empty:
                continue
                
            # Simple moving averages
            data['sma20'] = data['close'].rolling(20).mean()
            data['sma50'] = data['close'].rolling(50).mean()
            data['sma200'] = data['close'].rolling(200).mean()
            
            # RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # Store results
            latest = data.iloc[-1]
            indicators['moving_averages'][index_name] = {
                'price_vs_sma20': 'above' if latest['close'] > latest['sma20'] else 'below',
                'price_vs_sma50': 'above' if latest['close'] > latest['sma50'] else 'below',
                'price_vs_sma200': 'above' if latest['close'] > latest['sma200'] else 'below',
                'sma20_vs_sma50': 'above' if latest['sma20'] > latest['sma50'] else 'below'
            }
            
            indicators['oscillators'][index_name] = {
                'rsi': latest['rsi'],
                'rsi_zone': 'overbought' if latest['rsi'] > 70 else 
                           ('oversold' if latest['rsi'] < 30 else 'neutral')
            }
        
        # Analyze market breadth (simplified)
        bullish_indicators = 0
        total_indicators = 0
        
        for index, ma_data in indicators['moving_averages'].items():
            if ma_data['price_vs_sma50'] == 'above':
                bullish_indicators += 1
            total_indicators += 1
            
            if ma_data['price_vs_sma200'] == 'above':
                bullish_indicators += 1
            total_indicators += 1
        
        for index, osc_data in indicators['oscillators'].items():
            if osc_data['rsi_zone'] != 'overbought':  # Not being overbought is bullish
                bullish_indicators += 1
            total_indicators += 1
        
        if total_indicators > 0:
            breadth_score = bullish_indicators / total_indicators
            indicators['breadth'] = {
                'bullish_indicator_percent': breadth_score,
                'interpretation': 'bullish' if breadth_score > 0.7 else 
                                 ('bearish' if breadth_score < 0.3 else 'neutral')
            }
        else:
            indicators['breadth'] = {
                'bullish_indicator_percent': 0.5,
                'interpretation': 'neutral'
            }
            
        return indicators
    
    def _analyze_market_sentiment(self) -> Dict[str, Any]:
        """
        Analyze market sentiment based on news, social media, and other sources.
        
        Returns:
            dict: Sentiment analysis results.
        """
        # This would be implemented with actual sentiment analysis in a production system
        
        # Mock sentiment data for demonstration
        sentiment = {
            'news': {
                'score': np.random.uniform(-1, 1),
                'volume': np.random.randint(50, 200)
            },
            'social_media': {
                'score': np.random.uniform(-1, 1),
                'volume': np.random.randint(100, 500)
            },
            'analyst_ratings': {
                'bullish_percent': np.random.uniform(0.3, 0.7),
                'volume': np.random.randint(10, 50)
            }
        }
        
        # Calculate composite sentiment
        news_contrib = sentiment['news']['score'] * 0.4
        social_contrib = sentiment['social_media']['score'] * 0.3
        analyst_contrib = (sentiment['analyst_ratings']['bullish_percent'] - 0.5) * 2 * 0.3
        
        composite_score = news_contrib + social_contrib + analyst_contrib
        
        # Determine sentiment level
        if composite_score > 0.3:
            sentiment_level = 'bullish'
        elif composite_score < -0.3:
            sentiment_level = 'bearish'
        else:
            sentiment_level = 'neutral'
            
        sentiment['composite'] = {
            'score': composite_score,
            'level': sentiment_level
        }
        
        return sentiment
    
    def _analyze_macroeconomic_data(self) -> Dict[str, Any]:
        """
        Analyze macroeconomic data to determine economic conditions.
        
        Returns:
            dict: Economic indicators and conditions.
        """
        # This would be implemented with actual economic data in a production system
        
        # Mock economic data
        economic_data = {
            'gdp_growth': np.random.uniform(-1.0, 4.0),
            'unemployment': np.random.uniform(3.0, 7.0),
            'inflation': np.random.uniform(1.0, 5.0),
            'interest_rates': {
                'fed_funds': np.random.uniform(0.0, 3.0),
                'ten_year': np.random.uniform(1.0, 4.0)
            },
            'yield_curve': {
                'ten_year_minus_two_year': np.random.uniform(-0.5, 2.0)
            }
        }
        
        # Interpret data
        gdp_interp = 'expanding' if economic_data['gdp_growth'] > 2.0 else 'contracting' if economic_data['gdp_growth'] < 0 else 'slow_growth'
        
        unemp_interp = 'low' if economic_data['unemployment'] < 4.0 else 'high' if economic_data['unemployment'] > 6.0 else 'moderate'
        
        inflation_interp = 'high' if economic_data['inflation'] > 3.0 else 'low' if economic_data['inflation'] < 1.5 else 'moderate'
        
        yield_curve_interp = 'normal' if economic_data['yield_curve']['ten_year_minus_two_year'] > 0.5 else 'flat' if economic_data['yield_curve']['ten_year_minus_two_year'] > 0 else 'inverted'
        
        # Overall economic condition
        if gdp_interp in ['expanding'] and unemp_interp in ['low', 'moderate'] and yield_curve_interp != 'inverted':
            overall_condition = 'strong'
        elif gdp_interp == 'contracting' or (yield_curve_interp == 'inverted' and unemp_interp == 'high'):
            overall_condition = 'weak'
        else:
            overall_condition = 'mixed'
            
        economic_data['interpretations'] = {
            'gdp': gdp_interp,
            'unemployment': unemp_interp,
            'inflation': inflation_interp,
            'yield_curve': yield_curve_interp,
            'overall': overall_condition
        }
        
        return economic_data
    
    def _analyze_market_volatility(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate and interpret market volatility.
        
        Args:
            market_data (dict): Market data for various assets.
            
        Returns:
            dict: Volatility metrics and interpretation.
        """
        volatility_window = self.config['volatility_calc_window']
        volatility_metrics = {}
        
        # Calculate volatility for equity indices
        for index_name, data in market_data.get('equities', {}).items():
            if not isinstance(data, pd.DataFrame) or data.empty:
                continue
                
            # Calculate historical volatility (annualized standard deviation of returns)
            returns = data['close'].pct_change().dropna()
            if len(returns) >= volatility_window:
                hist_vol = returns.rolling(volatility_window).std().iloc[-1] * np.sqrt(252) * 100  # Annualized and in percentage
                volatility_metrics[index_name] = hist_vol
        
        # Calculate average volatility
        if volatility_metrics:
            avg_volatility = sum(volatility_metrics.values()) / len(volatility_metrics)
            
            # Determine volatility level based on thresholds
            thresholds = self.config['volatility_thresholds']
            if avg_volatility < thresholds['low']:
                vol_level = 'low'
            elif avg_volatility > thresholds['high']:
                vol_level = 'high'
            else:
                vol_level = 'medium'
                
            # Return volatility analysis
            return {
                'metrics': volatility_metrics,
                'average': avg_volatility,
                'level': vol_level
            }
        
        # Default if no data is available
        return {
            'metrics': {},
            'average': 15.0,  # Historical average
            'level': 'medium'
        }
    
    def _analyze_market_trends(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market trends and determine trend strength.
        
        Args:
            market_data (dict): Market data for various assets.
            
        Returns:
            dict: Trend analysis and strength.
        """
        trend_window = self.config['trend_calc_window']
        trend_metrics = {}
        
        # Analyze trends for equity indices
        for index_name, data in market_data.get('equities', {}).items():
            if not isinstance(data, pd.DataFrame) or data.empty:
                continue
            
            # Calculate linear regression slope over the trend window
            if len(data) >= trend_window:
                y = data['close'].values[-trend_window:]
                x = np.arange(trend_window)
                slope, _, _, _, _ = np.polyfit(x, y, 1, full=True)
                
                # Normalize slope to percentage of price
                norm_slope = slope[0] / np.mean(y) * 100
                
                # Calculate R-squared to measure trend strength
                y_mean = np.mean(y)
                y_pred = slope[0] * x + slope[1]
                r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2)
                
                trend_metrics[index_name] = {
                    'slope': norm_slope,
                    'direction': 'up' if norm_slope > 0 else 'down',
                    'r_squared': r_squared
                }
        
        # Calculate average trend strength
        if trend_metrics:
            avg_r_squared = sum(m['r_squared'] for m in trend_metrics.values()) / len(trend_metrics)
            avg_slope = sum(m['slope'] for m in trend_metrics.values()) / len(trend_metrics)
            
            # Determine trend strength
            thresholds = self.config['trend_strength_thresholds']
            if avg_r_squared < thresholds['weak']:
                strength = 'weak'
            elif avg_r_squared > thresholds['strong']:
                strength = 'strong'
            else:
                strength = 'medium'
                
            # Determine overall direction
            direction = 'up' if avg_slope > 0 else 'down'
            
            return {
                'metrics': trend_metrics,
                'average_r_squared': avg_r_squared,
                'average_slope': avg_slope,
                'direction': direction,
                'level': strength
            }
        
        # Default if no data is available
        return {
            'metrics': {},
            'average_r_squared': 0.3,
            'average_slope': 0,
            'direction': 'neutral',
            'level': 'weak'
        }
    
    def _analyze_cross_asset_correlations(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze correlations between different asset classes.
        
        Args:
            market_data (dict): Market data for various assets.
            
        Returns:
            dict: Correlation analysis and interpretation.
        """
        # Extract price data for correlation analysis
        price_series = {}
        
        for asset_class, assets in market_data.items():
            for asset_name, data in assets.items():
                if isinstance(data, pd.DataFrame) and not data.empty and 'close' in data.columns:
                    price_series[f"{asset_class}_{asset_name}"] = data['close']
        
        # Create correlation matrix if we have enough data
        if len(price_series) > 1:
            df = pd.DataFrame(price_series)
            # Calculate returns for correlation
            returns_df = df.pct_change().dropna()
            corr_matrix = returns_df.corr()
            
            # Calculate average correlation
            corr_values = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_values.append(corr_matrix.iloc[i, j])
                    
            avg_correlation = sum(corr_values) / len(corr_values) if corr_values else 0
            
            # Determine correlation level
            if avg_correlation > 0.7:
                corr_level = 'high'
            elif avg_correlation < 0.3:
                corr_level = 'low'
            else:
                corr_level = 'medium'
                
            return {
                'average': avg_correlation,
                'level': corr_level,
                'interpretation': 'risk_on' if avg_correlation > 0.6 else 'diversified'
            }
        
        # Default if we don't have enough data
        return {
            'average': 0.5,
            'level': 'medium',
            'interpretation': 'normal'
        }
    
    def _determine_market_regime(self, 
                               technical_indicators: Dict[str, Any],
                               sentiment_analysis: Dict[str, Any],
                               volatility_analysis: Dict[str, Any],
                               trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the current market regime based on various analyses.
        
        Args:
            technical_indicators (dict): Technical analysis results
            sentiment_analysis (dict): Sentiment analysis results
            volatility_analysis (dict): Volatility analysis results
            trend_analysis (dict): Trend analysis results
            
        Returns:
            dict: Market regime classification and confidence
        """
        # Extract key signals
        sentiment = sentiment_analysis.get('composite', {}).get('level', 'neutral')
        volatility = volatility_analysis.get('level', 'medium')
        trend_strength = trend_analysis.get('level', 'medium')
        trend_direction = trend_analysis.get('direction', 'neutral')
        
        breadth = technical_indicators.get('breadth', {}).get('interpretation', 'neutral')
        
        # Determine base market regime
        if trend_direction == 'up' and sentiment == 'bullish' and breadth == 'bullish':
            base_regime = 'bull'
        elif trend_direction == 'down' and (sentiment == 'bearish' or breadth == 'bearish'):
            base_regime = 'bear'
        else:
            base_regime = 'neutral'
            
        # Refine with volatility
        if volatility == 'high':
            if base_regime == 'bull':
                regime = 'bull_volatile'
            elif base_regime == 'bear':
                regime = 'bear_volatile'
            else:
                regime = 'volatile_sideways'
        else:
            regime = base_regime
            
        # Calculate confidence based on agreement of signals
        signals = [
            sentiment == 'bullish',  # Bullish sentiment
            sentiment == 'bearish',  # Bearish sentiment
            trend_direction == 'up',  # Uptrend
            trend_direction == 'down',  # Downtrend
            breadth == 'bullish',  # Bullish breadth
            breadth == 'bearish',  # Bearish breadth
        ]
        
        # Count positive and negative signals
        positive_signals = sum([signals[0], signals[2], signals[4]])
        negative_signals = sum([signals[1], signals[3], signals[5]])
        
        # Higher confidence when signals agree
        if positive_signals >= 2 and negative_signals == 0:
            confidence = 'high'
        elif negative_signals >= 2 and positive_signals == 0:
            confidence = 'high'
        elif positive_signals == 0 and negative_signals == 0:
            confidence = 'low'
        else:
            confidence = 'medium'
            
        return {
            'current': regime,
            'confidence': confidence,
            'primary_drivers': {
                'sentiment': sentiment,
                'trend': trend_direction,
                'volatility': volatility
            }
        }
            
    def _generate_neutral_context(self) -> Dict[str, Any]:
        """
        Generate a neutral market context when data is unavailable.
        
        Returns:
            dict: Neutral market context.
        """
        return {
            'market_regime': {
                'current': 'neutral',
                'confidence': 'low',
                'primary_drivers': {
                    'sentiment': 'neutral',
                    'trend': 'neutral',
                    'volatility': 'medium'
                }
            },
            'technical_indicators': {
                'moving_averages': {},
                'oscillators': {},
                'volatility': {},
                'breadth': {
                    'bullish_indicator_percent': 0.5,
                    'interpretation': 'neutral'
                }
            },
            'sentiment': {
                'composite': {
                    'score': 0,
                    'level': 'neutral'
                }
            },
            'macroeconomic': {
                'interpretations': {
                    'overall': 'mixed'
                }
            },
            'market_volatility': {
                'average': 15.0,
                'level': 'medium'
            },
            'trend_strength': {
                'direction': 'neutral',
                'level': 'weak'
            },
            'correlations': {
                'level': 'medium',
                'interpretation': 'normal'
            },
            'timestamp': datetime.now().isoformat()
        }
        
    def get_historical_context(self, days=7) -> List[Dict[str, Any]]:
        """
        Get historical market context data.
        
        Args:
            days (int): Number of days of history to retrieve
            
        Returns:
            list: Historical market contexts
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_time.isoformat()
        
        return [
            item for item in self.context_history
            if item['timestamp'] > cutoff_str
        ]
    
    def get_context_analysis(self) -> Dict[str, Any]:
        """
        Analyze changes in market context over time.
        
        Returns:
            dict: Analysis of market context evolution
        """
        if len(self.context_history) < 2:
            return {"status": "insufficient_data"}
            
        # Extract regime changes
        regimes = []
        for item in self.context_history:
            context = item['context']
            timestamp = item['timestamp']
            regime = context.get('market_regime', {}).get('current', 'unknown')
            regimes.append((timestamp, regime))
            
        # Count transitions
        transitions = 0
        for i in range(1, len(regimes)):
            if regimes[i][1] != regimes[i-1][1]:
                transitions += 1
                
        # Get most recent metrics
        latest_context = self.current_context
        latest_volatility = latest_context.get('market_volatility', {}).get('level', 'medium')
        latest_sentiment = latest_context.get('sentiment', {}).get('composite', {}).get('level', 'neutral')
        latest_regime = latest_context.get('market_regime', {}).get('current', 'neutral')
        
        # Create analysis
        return {
            "regime_transitions": transitions,
            "current_regime": latest_regime,
            "regime_stability": "unstable" if transitions > len(self.context_history) / 10 else "stable",
            "current_volatility": latest_volatility,
            "current_sentiment": latest_sentiment,
            "days_covered": len(self.context_history)
        } 