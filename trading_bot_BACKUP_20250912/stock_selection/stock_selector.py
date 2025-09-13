import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .sentiment_analyzer import SentimentAnalyzer
from .stock_scorer import StockScorer

logger = logging.getLogger(__name__)

class StockSelector:
    """
    Selects stocks for trading based on multiple criteria including:
    - News sentiment analysis
    - Technical indicators
    - Volume analysis
    - Custom filters
    """
    
    def __init__(self, data_provider=None):
        """
        Initialize the stock selector
        
        Args:
            data_provider: Provider for market data
        """
        self.scorer = StockScorer(data_provider)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.data_provider = data_provider
        logger.info("StockSelector initialized")
    
    def select_stocks(self, 
                      universe: List[str], 
                      min_score: float = 0.65,
                      max_stocks: int = 5,
                      filters: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Select stocks from the universe based on scoring and filtering
        
        Args:
            universe: List of ticker symbols to consider
            min_score: Minimum overall score to include stock
            max_stocks: Maximum number of stocks to select
            filters: Optional dictionary of filters to apply
            
        Returns:
            DataFrame with selected stocks and their scores
        """
        if not universe:
            logger.warning("Empty universe provided")
            return pd.DataFrame()
        
        logger.info(f"Selecting from {len(universe)} stocks with min_score={min_score}")
        
        # Score all stocks in universe
        scores_df = self.scorer.score_stocks(universe)
        
        if scores_df.empty:
            logger.warning("No scores available")
            return pd.DataFrame()
        
        # Apply minimum score filter
        filtered_df = scores_df[scores_df['overall_score'] >= min_score]
        
        # Apply additional filters if provided
        if filters:
            filtered_df = self._apply_filters(filtered_df, filters)
        
        # Return top N stocks
        result = filtered_df.head(max_stocks) if not filtered_df.empty else pd.DataFrame()
        
        logger.info(f"Selected {len(result)} stocks from {len(universe)} in universe")
        return result
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply custom filters to the stock DataFrame
        
        Args:
            df: DataFrame with stock scores
            filters: Dictionary of filters to apply
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # Apply sentiment filter
        if 'min_sentiment' in filters:
            min_sentiment = filters['min_sentiment']
            filtered_df = filtered_df[filtered_df['sentiment_score'] >= min_sentiment]
            logger.debug(f"Applied sentiment filter: {len(filtered_df)} stocks remaining")
        
        # Apply technical filter
        if 'min_technical' in filters:
            min_technical = filters['min_technical']
            filtered_df = filtered_df[filtered_df['technical_score'] >= min_technical]
            logger.debug(f"Applied technical filter: {len(filtered_df)} stocks remaining")
        
        # Apply volume filter
        if 'min_volume' in filters:
            min_volume = filters['min_volume']
            filtered_df = filtered_df[filtered_df['volume_score'] >= min_volume]
            logger.debug(f"Applied volume filter: {len(filtered_df)} stocks remaining")
        
        # Filter by recommendation
        if 'recommendation' in filters:
            recommendations = filters['recommendation']
            if isinstance(recommendations, str):
                recommendations = [recommendations]
            filtered_df = filtered_df[filtered_df['recommendation'].isin(recommendations)]
            logger.debug(f"Applied recommendation filter: {len(filtered_df)} stocks remaining")
        
        # Custom filter: minimum news count
        if 'min_news_count' in filters:
            min_news = filters['min_news_count']
            filtered_df = filtered_df[filtered_df['news_count'] >= min_news]
            logger.debug(f"Applied news count filter: {len(filtered_df)} stocks remaining")
        
        return filtered_df
    
    def select_stocks_by_sector(self, 
                               universe: List[str],
                               sector_allocation: Dict[str, float],
                               min_score: float = 0.6,
                               max_stocks_per_sector: int = 3) -> pd.DataFrame:
        """
        Select stocks by sector with specific allocation targets
        
        Args:
            universe: List of ticker symbols to consider
            sector_allocation: Dictionary mapping sectors to their target allocation percentages
            min_score: Minimum overall score to include stock
            max_stocks_per_sector: Maximum number of stocks per sector
            
        Returns:
            DataFrame with selected stocks and their scores
        """
        if not universe or not sector_allocation:
            logger.warning("Empty universe or sector allocation provided")
            return pd.DataFrame()
        
        if self.data_provider is None:
            logger.error("Data provider required for sector-based selection")
            return pd.DataFrame()
        
        logger.info(f"Selecting stocks by sector from {len(universe)} stocks")
        
        # Get sector information for all tickers
        sectors = {}
        try:
            for ticker in universe:
                sector = self.data_provider.get_sector(ticker)
                if sector:
                    if sector not in sectors:
                        sectors[sector] = []
                    sectors[sector].append(ticker)
        except Exception as e:
            logger.error(f"Error getting sector information: {str(e)}")
            return pd.DataFrame()
        
        # Score all stocks in universe
        scores_df = self.scorer.score_stocks(universe)
        
        if scores_df.empty:
            logger.warning("No scores available")
            return pd.DataFrame()
        
        # Apply minimum score filter
        scores_df = scores_df[scores_df['overall_score'] >= min_score]
        
        # Select top stocks for each sector
        selected_stocks = []
        for sector, allocation in sector_allocation.items():
            if sector in sectors and sectors[sector]:
                # Get tickers for this sector
                sector_tickers = sectors[sector]
                
                # Get scores for sector tickers
                sector_scores = scores_df[scores_df['ticker'].isin(sector_tickers)]
                
                # Select top stocks for this sector
                top_sector_stocks = sector_scores.head(max_stocks_per_sector)
                
                # Add sector and allocation information
                top_sector_stocks = top_sector_stocks.copy()
                top_sector_stocks['sector'] = sector
                top_sector_stocks['allocation'] = allocation
                
                selected_stocks.append(top_sector_stocks)
                
                logger.info(f"Selected {len(top_sector_stocks)} stocks from {sector} sector")
            else:
                logger.warning(f"No stocks found for sector: {sector}")
        
        # Combine all selected stocks
        if selected_stocks:
            result = pd.concat(selected_stocks)
            return result
        else:
            logger.warning("No stocks selected after sector filtering")
            return pd.DataFrame()
    
    def generate_portfolio(self, 
                          universe: List[str], 
                          total_capital: float,
                          max_stocks: int = 10,
                          sector_constraints: bool = True,
                          risk_profile: str = 'moderate') -> pd.DataFrame:
        """
        Generate a complete portfolio with allocation percentages
        
        Args:
            universe: List of ticker symbols to consider
            total_capital: Total capital to allocate
            max_stocks: Maximum number of stocks in portfolio
            sector_constraints: Whether to apply sector diversification constraints
            risk_profile: Risk profile ('conservative', 'moderate', 'aggressive')
            
        Returns:
            DataFrame with selected stocks, allocations, and dollar amounts
        """
        if not universe:
            logger.warning("Empty universe provided")
            return pd.DataFrame()
        
        logger.info(f"Generating portfolio from {len(universe)} stocks with {total_capital} capital")
        
        # Set min score based on risk profile
        min_scores = {
            'conservative': 0.7,
            'moderate': 0.65,
            'aggressive': 0.6
        }
        min_score = min_scores.get(risk_profile, 0.65)
        
        # Select stocks
        if sector_constraints and self.data_provider is not None:
            # For sector-constrained portfolio
            # Define sector allocation based on risk profile
            sector_allocations = self._get_sector_allocation(risk_profile)
            selected = self.select_stocks_by_sector(
                universe, 
                sector_allocations, 
                min_score=min_score,
                max_stocks_per_sector=max(1, max_stocks // len(sector_allocations))
            )
        else:
            # For unconstrained portfolio
            selected = self.select_stocks(
                universe,
                min_score=min_score,
                max_stocks=max_stocks
            )
            
            # Add equal allocation
            if not selected.empty:
                allocation_pct = 100.0 / len(selected)
                selected['allocation'] = allocation_pct
        
        # Calculate dollar amounts
        if not selected.empty:
            selected['dollar_amount'] = (selected['allocation'] / 100.0) * total_capital
            
            # Add number of shares (if price data available)
            if self.data_provider is not None:
                prices = []
                for ticker in selected['ticker']:
                    try:
                        price = self.data_provider.get_current_price(ticker)
                        prices.append(price if price > 0 else None)
                    except:
                        prices.append(None)
                
                selected['current_price'] = prices
                selected['shares'] = selected.apply(
                    lambda row: int(row['dollar_amount'] / row['current_price']) 
                    if row['current_price'] is not None and row['current_price'] > 0 else 0, 
                    axis=1
                )
        
        logger.info(f"Generated portfolio with {len(selected)} stocks")
        return selected
    
    def _get_sector_allocation(self, risk_profile: str) -> Dict[str, float]:
        """
        Get sector allocation based on risk profile
        
        Args:
            risk_profile: Risk profile ('conservative', 'moderate', 'aggressive')
            
        Returns:
            Dictionary with sector allocations
        """
        if risk_profile == 'conservative':
            return {
                'Utilities': 15.0,
                'Consumer Staples': 15.0,
                'Healthcare': 15.0,
                'Financial Services': 15.0,
                'Communication Services': 10.0,
                'Information Technology': 10.0,
                'Industrials': 10.0,
                'Consumer Discretionary': 5.0,
                'Materials': 5.0
            }
        elif risk_profile == 'aggressive':
            return {
                'Information Technology': 25.0,
                'Consumer Discretionary': 15.0,
                'Communication Services': 15.0,
                'Healthcare': 10.0,
                'Industrials': 10.0,
                'Financial Services': 10.0,
                'Materials': 5.0,
                'Energy': 5.0,
                'Consumer Staples': 5.0
            }
        else:  # moderate
            return {
                'Information Technology': 20.0,
                'Healthcare': 15.0,
                'Financial Services': 15.0,
                'Consumer Discretionary': 10.0,
                'Communication Services': 10.0,
                'Industrials': 10.0,
                'Consumer Staples': 10.0,
                'Materials': 5.0,
                'Utilities': 5.0
            }
    
    def rebalance_portfolio(self, 
                           current_portfolio: pd.DataFrame,
                           universe: List[str],
                           total_capital: float,
                           max_turnover_pct: float = 30.0,
                           min_position_pct: float = 5.0) -> Dict[str, pd.DataFrame]:
        """
        Rebalance an existing portfolio with minimal turnover
        
        Args:
            current_portfolio: DataFrame with current holdings
            universe: List of ticker symbols to consider
            total_capital: Total capital to allocate
            max_turnover_pct: Maximum percentage of portfolio to turn over
            min_position_pct: Minimum position size as percentage
            
        Returns:
            Dictionary with actions: 'hold', 'buy', 'sell'
        """
        if current_portfolio.empty or not universe:
            logger.warning("Empty portfolio or universe provided")
            return {'hold': pd.DataFrame(), 'buy': pd.DataFrame(), 'sell': pd.DataFrame()}
        
        logger.info(f"Rebalancing portfolio with {len(current_portfolio)} current positions")
        
        # Current holdings
        current_holdings = set(current_portfolio['ticker'].tolist())
        
        # Generate ideal new portfolio
        new_portfolio = self.generate_portfolio(
            universe, 
            total_capital,
            max_stocks=min(15, max(5, len(current_holdings) + 3))  # Allow for some growth
        )
        
        if new_portfolio.empty:
            logger.warning("Could not generate new portfolio")
            return {'hold': current_portfolio, 'buy': pd.DataFrame(), 'sell': pd.DataFrame()}
        
        # New holdings
        new_holdings = set(new_portfolio['ticker'].tolist())
        
        # Holdings to keep, add, remove
        keep_holdings = current_holdings.intersection(new_holdings)
        add_holdings = new_holdings.difference(current_holdings)
        remove_holdings = current_holdings.difference(new_holdings)
        
        # Calculate turnover
        current_value = current_portfolio['dollar_amount'].sum() if 'dollar_amount' in current_portfolio.columns else total_capital
        
        # Positions to remove
        sell_df = current_portfolio[current_portfolio['ticker'].isin(remove_holdings)].copy()
        sell_value = sell_df['dollar_amount'].sum() if not sell_df.empty and 'dollar_amount' in sell_df.columns else 0
        
        # Check if turnover exceeds limit
        if sell_value / current_value * 100 > max_turnover_pct:
            # Sort by score to keep better positions
            if 'overall_score' in current_portfolio.columns:
                sell_candidates = current_portfolio[current_portfolio['ticker'].isin(remove_holdings)].sort_values('overall_score')
            else:
                sell_candidates = current_portfolio[current_portfolio['ticker'].isin(remove_holdings)]
            
            # Sell only up to max turnover
            sell_value = 0
            sell_tickers = []
            for _, row in sell_candidates.iterrows():
                sell_value += row['dollar_amount']
                sell_tickers.append(row['ticker'])
                if sell_value / current_value * 100 >= max_turnover_pct:
                    break
            
            # Update holdings
            remove_holdings = set(sell_tickers)
            keep_holdings = current_holdings.difference(remove_holdings)
        
        # Calculate positions to hold
        hold_df = current_portfolio[current_portfolio['ticker'].isin(keep_holdings)].copy()
        
        # Calculate positions to buy
        buy_df = new_portfolio[new_portfolio['ticker'].isin(add_holdings)].copy()
        
        # Recalculate allocations for buys based on available capital
        available_capital = total_capital
        if not hold_df.empty and 'dollar_amount' in hold_df.columns:
            available_capital -= hold_df['dollar_amount'].sum()
        
        if not buy_df.empty and available_capital > 0:
            # Allocate available capital proportionally
            buy_allocation_total = buy_df['allocation'].sum()
            if buy_allocation_total > 0:
                buy_df['allocation'] = buy_df['allocation'] / buy_allocation_total * 100
                buy_df['dollar_amount'] = buy_df['allocation'] / 100 * available_capital
                
                # Remove positions that would be too small
                min_position_amount = total_capital * min_position_pct / 100
                buy_df = buy_df[buy_df['dollar_amount'] >= min_position_amount]
                
                # Recalculate shares if price data available
                if 'current_price' in buy_df.columns:
                    buy_df['shares'] = buy_df.apply(
                        lambda row: int(row['dollar_amount'] / row['current_price']) 
                        if row['current_price'] is not None and row['current_price'] > 0 else 0, 
                        axis=1
                    )
        
        # Calculate positions to sell
        sell_df = current_portfolio[current_portfolio['ticker'].isin(remove_holdings)].copy()
        
        # Log rebalance actions
        logger.info(f"Rebalance actions: hold={len(hold_df)}, buy={len(buy_df)}, sell={len(sell_df)}")
        
        return {
            'hold': hold_df,
            'buy': buy_df,
            'sell': sell_df
        } 