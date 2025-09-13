import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import talib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class StockScorer:
    """
    Scores stocks based on technical indicators, sentiment analysis, and volume patterns.
    Used for ranking stocks for potential trading opportunities.
    """
    
    def __init__(self, data_provider):
        """
        Initialize the stock scorer with a data provider
        
        Args:
            data_provider: Object that provides historical price and volume data
        """
        self.data_provider = data_provider
        logger.info("StockScorer initialized")
    
    def score_stocks(self, 
                    tickers: List[str], 
                    sentiment_data: Optional[Dict[str, Dict[str, Any]]] = None,
                    scoring_weights: Optional[Dict[str, float]] = None,
                    lookback_days: int = 100) -> pd.DataFrame:
        """
        Score a list of stocks based on technical, sentiment, and volume metrics
        
        Args:
            tickers: List of ticker symbols to score
            sentiment_data: Optional dictionary of sentiment scores by ticker
            scoring_weights: Optional dictionary of scoring weights for different factors
            lookback_days: Number of days of historical data to use
            
        Returns:
            DataFrame with tickers and their scores
        """
        if not tickers:
            logger.warning("No tickers provided for scoring")
            return pd.DataFrame()
        
        # Default scoring weights if not provided
        if not scoring_weights:
            scoring_weights = {
                'trend': 0.20,
                'momentum': 0.15,
                'volatility': 0.10,
                'volume': 0.15,
                'reversal': 0.10,
                'sentiment': 0.20,
                'fundamentals': 0.10
            }
        
        # Normalize weights
        total_weight = sum(scoring_weights.values())
        scoring_weights = {k: v/total_weight for k, v in scoring_weights.items()}
        
        # Calculate end date (today) and start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Final scores dataframe
        scores_df = pd.DataFrame(index=tickers)
        
        # Score each ticker
        for ticker in tickers:
            try:
                # Get historical data
                hist_data = self.data_provider.get_historical_data(
                    ticker, 
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'), 
                    interval='1d'
                )
                
                if hist_data is None or hist_data.empty or len(hist_data) < 20:
                    logger.warning(f"Insufficient data for {ticker}, skipping")
                    continue
                
                # Calculate individual scores
                technical_scores = self._calculate_technical_scores(hist_data)
                volume_score = self._calculate_volume_score(hist_data)
                
                # Add sentiment score if available
                sentiment_score = 0.5  # Neutral default
                if sentiment_data and ticker in sentiment_data:
                    sentiment_score = sentiment_data[ticker].get('sentiment_score', 0.5)
                
                # Get fundamental score if available through data provider
                fundamental_score = 0.5  # Neutral default
                try:
                    fundamental_data = self.data_provider.get_fundamental_data(ticker)
                    if fundamental_data:
                        fundamental_score = self._calculate_fundamental_score(fundamental_data)
                except:
                    logger.debug(f"Could not retrieve fundamental data for {ticker}")
                
                # Combine all scores with weights
                final_score = (
                    technical_scores['trend'] * scoring_weights.get('trend', 0.0) +
                    technical_scores['momentum'] * scoring_weights.get('momentum', 0.0) +
                    technical_scores['volatility'] * scoring_weights.get('volatility', 0.0) +
                    technical_scores['reversal'] * scoring_weights.get('reversal', 0.0) +
                    volume_score * scoring_weights.get('volume', 0.0) +
                    sentiment_score * scoring_weights.get('sentiment', 0.0) +
                    fundamental_score * scoring_weights.get('fundamentals', 0.0)
                )
                
                # Store all scores in the dataframe
                scores_df.loc[ticker, 'total_score'] = final_score
                scores_df.loc[ticker, 'trend_score'] = technical_scores['trend']
                scores_df.loc[ticker, 'momentum_score'] = technical_scores['momentum']
                scores_df.loc[ticker, 'volatility_score'] = technical_scores['volatility']
                scores_df.loc[ticker, 'reversal_score'] = technical_scores['reversal']
                scores_df.loc[ticker, 'volume_score'] = volume_score
                scores_df.loc[ticker, 'sentiment_score'] = sentiment_score
                scores_df.loc[ticker, 'fundamental_score'] = fundamental_score
            
            except Exception as e:
                logger.error(f"Error scoring {ticker}: {str(e)}")
                continue
        
        # Sort by total score (descending)
        if not scores_df.empty:
            scores_df = scores_df.sort_values('total_score', ascending=False)
        
        return scores_df
    
    def _calculate_technical_scores(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate technical analysis scores
        
        Args:
            data: DataFrame with historical price data
            
        Returns:
            Dictionary with various technical scores
        """
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                logger.warning(f"Missing required column: {col}")
                return {
                    'trend': 0.5,
                    'momentum': 0.5,
                    'volatility': 0.5,
                    'reversal': 0.5
                }
        
        # Calculate trend indicators
        trend_score = self._calculate_trend_score(data)
        
        # Calculate momentum indicators
        momentum_score = self._calculate_momentum_score(data)
        
        # Calculate volatility indicators
        volatility_score = self._calculate_volatility_score(data)
        
        # Calculate reversal indicators
        reversal_score = self._calculate_reversal_score(data)
        
        return {
            'trend': trend_score,
            'momentum': momentum_score,
            'volatility': volatility_score,
            'reversal': reversal_score
        }
    
    def _calculate_trend_score(self, data: pd.DataFrame) -> float:
        """
        Calculate trend score based on moving averages and ADX
        
        Args:
            data: DataFrame with historical price data
            
        Returns:
            Trend score from 0 to 1
        """
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            
            # Calculate moving averages
            ma20 = talib.SMA(close, timeperiod=20)
            ma50 = talib.SMA(close, timeperiod=50)
            ma200 = talib.SMA(close, timeperiod=200)
            
            # Calculate ADX (trend strength)
            adx = talib.ADX(high, low, close, timeperiod=14)
            
            # Calculate MACD
            macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            
            # Price relative to moving averages
            price_vs_ma20 = close[-1] / ma20[-1] if not np.isnan(ma20[-1]) and ma20[-1] != 0 else 1
            price_vs_ma50 = close[-1] / ma50[-1] if not np.isnan(ma50[-1]) and ma50[-1] != 0 else 1
            price_vs_ma200 = close[-1] / ma200[-1] if not np.isnan(ma200[-1]) and ma200[-1] != 0 else 1
            
            # MA crossovers
            ma20_vs_50 = ma20[-1] / ma50[-1] if not np.isnan(ma50[-1]) and ma50[-1] != 0 else 1
            ma50_vs_200 = ma50[-1] / ma200[-1] if not np.isnan(ma200[-1]) and ma200[-1] != 0 else 1
            
            # MACD signal
            macd_signal_value = 0.5
            if not np.isnan(macd[-1]) and not np.isnan(macd_signal[-1]):
                if macd[-1] > macd_signal[-1]:
                    macd_signal_value = 0.7
                elif macd[-1] < macd_signal[-1]:
                    macd_signal_value = 0.3
            
            # ADX strength
            adx_strength = 0.5
            if not np.isnan(adx[-1]):
                if adx[-1] < 20:  # Weak trend
                    adx_strength = 0.4
                elif adx[-1] < 40:  # Moderate trend
                    adx_strength = 0.6
                else:  # Strong trend
                    adx_strength = 0.8
            
            # Calculate trend direction
            trend_direction = 0.5
            if (price_vs_ma20 > 1 and price_vs_ma50 > 1 and price_vs_ma200 > 1):
                trend_direction = 0.9  # Strong uptrend
            elif (price_vs_ma20 > 1 and price_vs_ma50 > 1):
                trend_direction = 0.7  # Moderate uptrend
            elif (price_vs_ma20 < 1 and price_vs_ma50 < 1 and price_vs_ma200 < 1):
                trend_direction = 0.1  # Strong downtrend
            elif (price_vs_ma20 < 1 and price_vs_ma50 < 1):
                trend_direction = 0.3  # Moderate downtrend
            
            # Combine factors for overall trend score
            trend_score = (
                trend_direction * 0.5 +
                adx_strength * 0.2 +
                macd_signal_value * 0.3
            )
            
            return trend_score
            
        except Exception as e:
            logger.error(f"Error calculating trend score: {str(e)}")
            return 0.5
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """
        Calculate momentum score based on RSI, Stochastic, and rate of change
        
        Args:
            data: DataFrame with historical price data
            
        Returns:
            Momentum score from 0 to 1
        """
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            
            # Calculate RSI
            rsi = talib.RSI(close, timeperiod=14)
            
            # Calculate Stochastic
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            
            # Calculate Rate of Change
            roc = talib.ROC(close, timeperiod=10)
            
            # Calculate Momentum
            mom = talib.MOM(close, timeperiod=10)
            
            # RSI score (favor mid-range to higher RSI, avoid overbought)
            rsi_score = 0.5
            if not np.isnan(rsi[-1]):
                if rsi[-1] < 30:  # Oversold
                    rsi_score = 0.3
                elif rsi[-1] < 50:  # Rising from oversold
                    rsi_score = 0.7
                elif rsi[-1] < 70:  # Bullish momentum
                    rsi_score = 0.8
                else:  # Overbought
                    rsi_score = 0.4
            
            # Stochastic score
            stoch_score = 0.5
            if not np.isnan(slowk[-1]) and not np.isnan(slowd[-1]):
                if slowk[-1] < 20 and slowd[-1] < 20:  # Oversold
                    stoch_score = 0.3
                elif slowk[-1] > slowd[-1] and slowk[-1] < 80:  # Bullish crossover
                    stoch_score = 0.8
                elif slowk[-1] > 80 and slowd[-1] > 80:  # Overbought
                    stoch_score = 0.4
            
            # ROC score
            roc_score = 0.5
            if not np.isnan(roc[-1]):
                if roc[-1] > 5:  # Strong positive momentum
                    roc_score = 0.8
                elif roc[-1] > 0:  # Positive momentum
                    roc_score = 0.7
                elif roc[-1] > -5:  # Weak negative momentum
                    roc_score = 0.4
                else:  # Strong negative momentum
                    roc_score = 0.2
            
            # Momentum direction
            mom_score = 0.5
            if not np.isnan(mom[-1]):
                if mom[-1] > 0:
                    mom_score = 0.7
                else:
                    mom_score = 0.3
            
            # Combine all momentum factors
            momentum_score = (
                rsi_score * 0.3 +
                stoch_score * 0.2 +
                roc_score * 0.3 +
                mom_score * 0.2
            )
            
            return momentum_score
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {str(e)}")
            return 0.5
    
    def _calculate_volatility_score(self, data: pd.DataFrame) -> float:
        """
        Calculate volatility score based on ATR, Bollinger Bands, and historical volatility
        
        Args:
            data: DataFrame with historical price data
            
        Returns:
            Volatility score from 0 to 1 (higher means more favorable volatility)
        """
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            
            # Calculate ATR
            atr = talib.ATR(high, low, close, timeperiod=14)
            
            # Calculate Bollinger Bands
            upper, middle, lower = talib.BBANDS(close, timeperiod=20)
            
            # Calculate Historical Volatility (Standard Deviation)
            stddev = talib.STDDEV(close, timeperiod=20)
            
            # ATR relative to price (normalized)
            atr_ratio = atr[-1] / close[-1] if not np.isnan(atr[-1]) and close[-1] != 0 else 0
            
            # Bollinger Band Width
            bb_width = (upper[-1] - lower[-1]) / middle[-1] if not np.isnan(middle[-1]) and middle[-1] != 0 else 0
            
            # StdDev relative to price
            stddev_ratio = stddev[-1] / close[-1] if not np.isnan(stddev[-1]) and close[-1] != 0 else 0
            
            # Position within Bollinger Bands
            bb_position = 0.5
            if not np.isnan(upper[-1]) and not np.isnan(lower[-1]) and not np.isnan(close[-1]):
                bb_range = upper[-1] - lower[-1]
                if bb_range > 0:
                    bb_position = (close[-1] - lower[-1]) / bb_range
            
            # Volatility score calculation
            # We want moderate volatility - not too high, not too low
            # Normalize ATR ratio to favor moderate volatility
            atr_score = 0.5
            if atr_ratio > 0:
                if atr_ratio < 0.01:  # Very low volatility
                    atr_score = 0.3
                elif atr_ratio < 0.02:  # Low volatility
                    atr_score = 0.6
                elif atr_ratio < 0.03:  # Moderate volatility
                    atr_score = 0.8
                elif atr_ratio < 0.05:  # High volatility
                    atr_score = 0.7
                else:  # Very high volatility
                    atr_score = 0.4
            
            # BB width score
            bb_score = 0.5
            if bb_width > 0:
                if bb_width < 0.05:  # Very narrow bands
                    bb_score = 0.3  # Likely to expand soon
                elif bb_width < 0.1:  # Moderate width
                    bb_score = 0.7  # Good trading opportunities
                else:  # Wide bands
                    bb_score = 0.5  # Could indicate excessive volatility
            
            # BB position score
            position_score = 0.5
            if bb_position <= 0.2:  # Near lower band
                position_score = 0.7  # Potential bounce
            elif bb_position >= 0.8:  # Near upper band
                position_score = 0.3  # Potential drop
            else:  # Middle of the bands
                position_score = 0.5  # Neutral
            
            # Combine all volatility factors
            volatility_score = (
                atr_score * 0.3 +
                bb_score * 0.3 +
                position_score * 0.4
            )
            
            return volatility_score
            
        except Exception as e:
            logger.error(f"Error calculating volatility score: {str(e)}")
            return 0.5
    
    def _calculate_reversal_score(self, data: pd.DataFrame) -> float:
        """
        Calculate reversal potential score based on candlestick patterns and divergences
        
        Args:
            data: DataFrame with historical price data
            
        Returns:
            Reversal score from 0 to 1
        """
        try:
            close = data['close'].values
            open_price = data['open'].values
            high = data['high'].values
            low = data['low'].values
            
            # Calculate candlestick patterns
            doji = talib.CDLDOJI(open_price, high, low, close)
            hammer = talib.CDLHAMMER(open_price, high, low, close)
            engulfing = talib.CDLENGULFING(open_price, high, low, close)
            morning_star = talib.CDLMORNINGSTAR(open_price, high, low, close)
            evening_star = talib.CDLEVENINGSTAR(open_price, high, low, close)
            
            # Check for patterns in last 3 days
            bullish_patterns = sum([
                1 if hammer[-i] > 0 else 0 for i in range(1, 4)
            ]) + sum([
                1 if engulfing[-i] > 0 else 0 for i in range(1, 4)
            ]) + sum([
                1 if morning_star[-i] > 0 else 0 for i in range(1, 4)
            ])
            
            bearish_patterns = sum([
                1 if doji[-i] < 0 else 0 for i in range(1, 4)
            ]) + sum([
                1 if engulfing[-i] < 0 else 0 for i in range(1, 4)
            ]) + sum([
                1 if evening_star[-i] < 0 else 0 for i in range(1, 4)
            ])
            
            # Calculate RSI for divergence check
            rsi = talib.RSI(close, timeperiod=14)
            
            # Check for bullish divergence (price lower low, RSI higher low)
            bullish_divergence = False
            if len(close) > 20 and len(rsi) > 20:
                price_lower_low = close[-1] < min(close[-20:-1])
                rsi_higher_low = rsi[-1] > min(rsi[-20:-1])
                bullish_divergence = price_lower_low and rsi_higher_low
            
            # Check for bearish divergence (price higher high, RSI lower high)
            bearish_divergence = False
            if len(close) > 20 and len(rsi) > 20:
                price_higher_high = close[-1] > max(close[-20:-1])
                rsi_lower_high = rsi[-1] < max(rsi[-20:-1])
                bearish_divergence = price_higher_high and rsi_lower_high
            
            # Combine all reversal factors
            reversal_score = 0.5  # Neutral default
            
            # Adjust based on patterns and divergences
            if bullish_patterns > 0 or bullish_divergence:
                reversal_score = min(0.8, 0.5 + (bullish_patterns * 0.1) + (0.2 if bullish_divergence else 0))
            elif bearish_patterns > 0 or bearish_divergence:
                reversal_score = max(0.2, 0.5 - (bearish_patterns * 0.1) - (0.2 if bearish_divergence else 0))
            
            return reversal_score
            
        except Exception as e:
            logger.error(f"Error calculating reversal score: {str(e)}")
            return 0.5
    
    def _calculate_volume_score(self, data: pd.DataFrame) -> float:
        """
        Calculate volume analysis score
        
        Args:
            data: DataFrame with historical price data including volume
            
        Returns:
            Volume score from 0 to 1
        """
        try:
            if 'volume' not in data.columns:
                return 0.5
            
            close = data['close'].values
            volume = data['volume'].values
            
            # Calculate volume indicators
            volume_sma20 = talib.SMA(volume, timeperiod=20)
            
            # Calculate On-Balance Volume (OBV)
            obv = talib.OBV(close, volume)
            
            # Calculate Chaikin Money Flow
            cmf = talib.ADOSC(data['high'].values, data['low'].values, close, volume, fastperiod=3, slowperiod=10)
            
            # Recent volume relative to average
            rel_volume = volume[-1] / volume_sma20[-1] if not np.isnan(volume_sma20[-1]) and volume_sma20[-1] > 0 else 1.0
            
            # Volume trend (last 5 days vs previous 5 days)
            recent_vol_avg = np.mean(volume[-5:])
            prev_vol_avg = np.mean(volume[-10:-5])
            vol_trend = recent_vol_avg / prev_vol_avg if prev_vol_avg > 0 else 1.0
            
            # OBV trend
            obv_trend = 0.5
            if len(obv) > 10:
                obv_slope = (obv[-1] - obv[-10]) / 10
                if obv_slope > 0:
                    obv_trend = 0.7
                else:
                    obv_trend = 0.3
            
            # CMF signal
            cmf_signal = 0.5
            if not np.isnan(cmf[-1]):
                if cmf[-1] > 0.05:  # Strong buying pressure
                    cmf_signal = 0.8
                elif cmf[-1] > 0:  # Moderate buying pressure
                    cmf_signal = 0.7
                elif cmf[-1] > -0.05:  # Moderate selling pressure
                    cmf_signal = 0.4
                else:  # Strong selling pressure
                    cmf_signal = 0.2
            
            # Calculate price/volume correlation recently
            try:
                recent_price_changes = np.diff(close[-10:])
                recent_volume_changes = np.diff(volume[-10:])
                if len(recent_price_changes) > 1 and len(recent_volume_changes) > 1:
                    correlation = np.corrcoef(recent_price_changes, recent_volume_changes)[0, 1]
                    if not np.isnan(correlation):
                        correlation_score = 0.7 if correlation > 0 else 0.3
                    else:
                        correlation_score = 0.5
                else:
                    correlation_score = 0.5
            except:
                correlation_score = 0.5
            
            # Combine volume factors
            volume_score = (
                (0.7 if rel_volume > 1.5 else 0.5 if rel_volume > 0.8 else 0.3) * 0.2 +  # Recent volume
                (0.7 if vol_trend > 1.2 else 0.5 if vol_trend > 0.8 else 0.3) * 0.2 +  # Volume trend
                obv_trend * 0.2 +  # OBV trend
                cmf_signal * 0.3 +  # CMF
                correlation_score * 0.1  # Price/volume correlation
            )
            
            return volume_score
            
        except Exception as e:
            logger.error(f"Error calculating volume score: {str(e)}")
            return 0.5
    
    def _calculate_fundamental_score(self, fundamental_data: Dict[str, Any]) -> float:
        """
        Calculate score based on fundamental metrics
        
        Args:
            fundamental_data: Dictionary with fundamental data
            
        Returns:
            Fundamental score from 0 to 1
        """
        try:
            # Extract key fundamental metrics
            pe_ratio = fundamental_data.get('pe_ratio')
            eps_growth = fundamental_data.get('eps_growth')
            revenue_growth = fundamental_data.get('revenue_growth')
            profit_margin = fundamental_data.get('profit_margin')
            debt_to_equity = fundamental_data.get('debt_to_equity')
            current_ratio = fundamental_data.get('current_ratio')
            price_to_book = fundamental_data.get('price_to_book')
            return_on_equity = fundamental_data.get('return_on_equity')
            
            scores = []
            
            # PE Ratio (lower is better, but negative is bad)
            if pe_ratio is not None:
                if pe_ratio <= 0:
                    scores.append(0.3)  # Negative earnings
                elif pe_ratio < 10:
                    scores.append(0.9)  # Very attractive
                elif pe_ratio < 15:
                    scores.append(0.8)  # Attractive
                elif pe_ratio < 25:
                    scores.append(0.6)  # Fair
                elif pe_ratio < 50:
                    scores.append(0.4)  # Expensive
                else:
                    scores.append(0.2)  # Very expensive
            
            # EPS Growth
            if eps_growth is not None:
                if eps_growth > 30:
                    scores.append(0.9)  # Excellent growth
                elif eps_growth > 15:
                    scores.append(0.8)  # Strong growth
                elif eps_growth > 5:
                    scores.append(0.7)  # Good growth
                elif eps_growth > 0:
                    scores.append(0.6)  # Positive growth
                elif eps_growth > -10:
                    scores.append(0.4)  # Slight decline
                else:
                    scores.append(0.2)  # Significant decline
            
            # Revenue Growth
            if revenue_growth is not None:
                if revenue_growth > 30:
                    scores.append(0.9)  # Excellent growth
                elif revenue_growth > 15:
                    scores.append(0.8)  # Strong growth
                elif revenue_growth > 5:
                    scores.append(0.7)  # Good growth
                elif revenue_growth > 0:
                    scores.append(0.6)  # Positive growth
                elif revenue_growth > -10:
                    scores.append(0.4)  # Slight decline
                else:
                    scores.append(0.2)  # Significant decline
            
            # Profit Margin
            if profit_margin is not None:
                if profit_margin > 20:
                    scores.append(0.9)  # Excellent margin
                elif profit_margin > 10:
                    scores.append(0.8)  # Strong margin
                elif profit_margin > 5:
                    scores.append(0.7)  # Good margin
                elif profit_margin > 0:
                    scores.append(0.6)  # Positive margin
                else:
                    scores.append(0.3)  # Loss
            
            # Debt to Equity
            if debt_to_equity is not None:
                if debt_to_equity < 0.3:
                    scores.append(0.9)  # Very low debt
                elif debt_to_equity < 0.7:
                    scores.append(0.8)  # Low debt
                elif debt_to_equity < 1.2:
                    scores.append(0.7)  # Moderate debt
                elif debt_to_equity < 2:
                    scores.append(0.5)  # High debt
                else:
                    scores.append(0.3)  # Very high debt
            
            # Current Ratio
            if current_ratio is not None:
                if current_ratio > 2:
                    scores.append(0.9)  # Very liquid
                elif current_ratio > 1.5:
                    scores.append(0.8)  # Good liquidity
                elif current_ratio > 1:
                    scores.append(0.7)  # Adequate liquidity
                elif current_ratio > 0.8:
                    scores.append(0.5)  # Moderate liquidity concerns
                else:
                    scores.append(0.3)  # Liquidity concerns
            
            # Price to Book
            if price_to_book is not None:
                if price_to_book < 1:
                    scores.append(0.9)  # Trading below book value
                elif price_to_book < 2:
                    scores.append(0.8)  # Attractive
                elif price_to_book < 3:
                    scores.append(0.7)  # Reasonable
                elif price_to_book < 5:
                    scores.append(0.5)  # Fair
                else:
                    scores.append(0.3)  # Expensive
            
            # Return on Equity
            if return_on_equity is not None:
                if return_on_equity > 25:
                    scores.append(0.9)  # Excellent returns
                elif return_on_equity > 15:
                    scores.append(0.8)  # Strong returns
                elif return_on_equity > 10:
                    scores.append(0.7)  # Good returns
                elif return_on_equity > 5:
                    scores.append(0.6)  # Moderate returns
                elif return_on_equity > 0:
                    scores.append(0.5)  # Positive returns
                else:
                    scores.append(0.3)  # Negative returns
            
            # Calculate average score if we have enough data
            if len(scores) >= 3:
                return sum(scores) / len(scores)
            else:
                return 0.5  # Default score if insufficient data
                
        except Exception as e:
            logger.error(f"Error calculating fundamental score: {str(e)}")
            return 0.5
    
    def get_top_stocks(self, 
                      scored_stocks: pd.DataFrame, 
                      top_n: int = 10, 
                      min_score: float = 0.6) -> List[str]:
        """
        Get top N stocks with scores above the minimum threshold
        
        Args:
            scored_stocks: DataFrame with stock scores
            top_n: Number of top stocks to return
            min_score: Minimum score threshold
            
        Returns:
            List of top ticker symbols
        """
        if scored_stocks.empty:
            return []
        
        # Filter by minimum score
        filtered_stocks = scored_stocks[scored_stocks['total_score'] >= min_score]
        
        # If no stocks meet minimum score, return empty list
        if filtered_stocks.empty:
            return []
        
        # Sort by total score and get top N
        top_stocks = filtered_stocks.sort_values('total_score', ascending=False).head(top_n)
        
        return top_stocks.index.tolist()
    
    def explain_scores(self, ticker: str, scores: pd.Series) -> Dict[str, Any]:
        """
        Generate explanations for why a stock received its scores
        
        Args:
            ticker: Ticker symbol
            scores: Series with scores for the ticker
            
        Returns:
            Dictionary with score explanations
        """
        explanations = {}
        
        # Total score explanation
        total_score = scores.get('total_score', 0)
        if total_score >= 0.8:
            explanations['total'] = f"{ticker} has very strong overall metrics with a score of {total_score:.2f}"
        elif total_score >= 0.7:
            explanations['total'] = f"{ticker} shows strong potential with a score of {total_score:.2f}"
        elif total_score >= 0.6:
            explanations['total'] = f"{ticker} has above average metrics with a score of {total_score:.2f}"
        elif total_score >= 0.5:
            explanations['total'] = f"{ticker} has neutral overall metrics with a score of {total_score:.2f}"
        else:
            explanations['total'] = f"{ticker} shows weak overall metrics with a score of {total_score:.2f}"
        
        # Technical trend explanation
        trend_score = scores.get('trend_score', 0)
        if trend_score >= 0.8:
            explanations['trend'] = f"Very strong uptrend"
        elif trend_score >= 0.7:
            explanations['trend'] = f"Solid uptrend"
        elif trend_score >= 0.6:
            explanations['trend'] = f"Positive trend developing"
        elif trend_score >= 0.4:
            explanations['trend'] = f"Neutral trend"
        elif trend_score >= 0.3:
            explanations['trend'] = f"Negative trend"
        else:
            explanations['trend'] = f"Strong downtrend"
        
        # Momentum explanation
        momentum_score = scores.get('momentum_score', 0)
        if momentum_score >= 0.8:
            explanations['momentum'] = f"Very strong momentum"
        elif momentum_score >= 0.7:
            explanations['momentum'] = f"Good positive momentum"
        elif momentum_score >= 0.6:
            explanations['momentum'] = f"Moderately positive momentum"
        elif momentum_score >= 0.4:
            explanations['momentum'] = f"Neutral momentum"
        elif momentum_score >= 0.3:
            explanations['momentum'] = f"Weakening momentum"
        else:
            explanations['momentum'] = f"Negative momentum"
        
        # Volume explanation
        volume_score = scores.get('volume_score', 0)
        if volume_score >= 0.8:
            explanations['volume'] = f"Very strong volume patterns"
        elif volume_score >= 0.7:
            explanations['volume'] = f"Healthy volume supporting price action"
        elif volume_score >= 0.6:
            explanations['volume'] = f"Positive volume trends"
        elif volume_score >= 0.4:
            explanations['volume'] = f"Neutral volume patterns"
        elif volume_score >= 0.3:
            explanations['volume'] = f"Concerning volume trends"
        else:
            explanations['volume'] = f"Weak volume patterns"
        
        # Sentiment explanation
        sentiment_score = scores.get('sentiment_score', 0)
        if sentiment_score >= 0.8:
            explanations['sentiment'] = f"Very positive market sentiment"
        elif sentiment_score >= 0.7:
            explanations['sentiment'] = f"Positive market sentiment"
        elif sentiment_score >= 0.6:
            explanations['sentiment'] = f"Somewhat positive sentiment"
        elif sentiment_score >= 0.4:
            explanations['sentiment'] = f"Neutral market sentiment"
        elif sentiment_score >= 0.3:
            explanations['sentiment'] = f"Somewhat negative sentiment"
        else:
            explanations['sentiment'] = f"Negative market sentiment"
        
        return explanations 