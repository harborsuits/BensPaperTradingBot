#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sector Rotation Strategy

A sophisticated strategy for trading economic cycles through sector rotation.
This strategy identifies and exploits the tendency of different market sectors 
to outperform at different stages of the economic cycle.
"""

import logging
import numpy as np
import pandas as pd
import uuid
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import copy

from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.core.constants import TimeFrame, MarketType
from trading_bot.core.signals import Signal, SignalType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.position import Position, PositionStatus
from trading_bot.strategies_new.stocks.base.stocks_base_strategy import StocksBaseStrategy, StocksSession
from trading_bot.strategies_new.factory.registry import register_strategy

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="SectorRotationStrategy",
    market_type="stocks",
    description="A portfolio strategy that systematically allocates capital to different market sectors based on economic cycle analysis",
    timeframes=["1d", "1w", "1M"],
    parameters={
        "rotation_model": {"description": "Model for sector rotation (economic_cycle, momentum, relative_strength, adaptive)", "type": "string"},
        "rotation_frequency": {"description": "How often to rotate sectors (weekly, monthly, quarterly)", "type": "string"},
        "trade_sector_etfs": {"description": "Whether to trade sector ETFs or individual stocks", "type": "boolean"},
        "defensive_threshold": {"description": "Threshold for switching to defensive positioning", "type": "float"}
    }
)
class SectorRotationStrategy(StocksBaseStrategy):
    """
    Sector Rotation Strategy
    
    A portfolio strategy that systematically allocates capital to different market sectors
    based on economic cycle analysis, sector momentum, and relative strength.
    
    Features:
    - Economic cycle stage identification and forecasting
    - Sector relative performance analysis
    - Adaptive portfolio allocation based on sector outlook
    - Multiple rotation models (classic cycle, momentum, and combined)
    - Managed gradual rotation to reduce transaction costs
    - Defensive positioning during economic downturns
    """
    
    # Standard economic cycle sector relationships
    # Based on classical sector rotation theory
    ECONOMIC_CYCLE_SECTORS = {
        'early_recession': ['utilities', 'consumer_staples', 'healthcare'],
        'late_recession': ['financials', 'consumer_discretionary', 'technology'],
        'early_recovery': ['industrials', 'basic_materials', 'energy'],
        'late_recovery': ['energy', 'basic_materials', 'utilities'],
        'early_expansion': ['industrials', 'technology', 'financials'],
        'late_expansion': ['consumer_staples', 'healthcare', 'utilities'],
    }
    
    # Sector ETF mapping for easy trading
    SECTOR_ETFS = {
        'technology': 'XLK',
        'healthcare': 'XLV',
        'financials': 'XLF',
        'consumer_discretionary': 'XLY',
        'consumer_staples': 'XLP',
        'energy': 'XLE',
        'utilities': 'XLU',
        'real_estate': 'XLRE',
        'industrials': 'XLI',
        'basic_materials': 'XLB',
        'communication_services': 'XLC'
    }
    
    def __init__(self, session: StocksSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the Sector Rotation Strategy.
        
        Args:
            session: StocksSession for the specific symbol and timeframe
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Initialize the base strategy - note that for a sector rotation strategy,
        # the session symbol is typically a market index or benchmark ETF (e.g., SPY)
        super().__init__(session, data_pipeline, parameters)
        
        # Strategy-specific default parameters
        default_params = {
            # Rotation model parameters
            'rotation_model': 'combined',   # 'economic_cycle', 'momentum', 'relative_strength', or 'combined'
            'economic_cycle_weight': 0.4,    # Weight for economic cycle model (in combined model)
            'momentum_weight': 0.3,          # Weight for momentum model (in combined model)
            'relative_strength_weight': 0.3, # Weight for relative strength model (in combined model)
            
            # Economic indicators and forecasting
            'enable_economic_forecasting': True,     # Use economic data to forecast cycle stage
            'economic_data_sources': ['fed', 'bls'], # Sources for economic data
            'forecast_horizon_months': 3,            # Months ahead to forecast
            
            # Portfolio allocation parameters
            'max_sectors': 3,               # Maximum number of sectors to hold
            'sector_concentration': 'equal_weight',  # 'equal_weight', 'market_weight', or 'custom'
            'custom_sector_weights': {},     # If custom weights used
            'min_sector_allocation': 0.15,   # Minimum allocation to any sector (if included)
            'max_sector_allocation': 0.40,   # Maximum allocation to any sector
            'allow_short_sectors': False,    # Whether to allow shorting underperforming sectors
            
            # Execution parameters
            'rebalance_frequency': 'monthly',  # 'weekly', 'monthly', 'quarterly'
            'gradual_rotation': True,          # Gradually rotate instead of immediately
            'rotation_steps': 3,               # Number of steps for gradual rotation
            'trade_sector_etfs': True,         # Trade sector ETFs instead of individual stocks
            'stocks_per_sector': 5,            # Number of stocks per sector if not using ETFs
            'stock_selection_method': 'market_cap',  # How to select stocks within sectors
            
            # Risk management
            'max_drawdown_exit': 15.0,       # Maximum drawdown percentage before defensive positioning
            'volatility_cap': 20.0,          # Annualized volatility cap
            'correlation_threshold': 0.7,     # Correlation threshold for diversification
            'defensive_allocation': {         # Allocation during defensive positioning
                'cash': 0.50,
                'utilities': 0.25,
                'consumer_staples': 0.25
            },
            
            # Performance evaluation
            'benchmark_index': 'SPY',        # Benchmark for performance comparison
            'minimum_outperformance': 1.0,    # Required outperformance (%) to trigger rotation
            'evaluation_window': 60,          # Days to evaluate sector performance
            
            # Advanced parameters
            'adapt_to_market_regime': True,   # Adapt rotation model to market regime
            'use_alternative_data': False,    # Use alternative data sources
            'alternative_data_sources': [],   # List of alternative data sources
            'include_international': False,   # Include international sectors
            
            # External data
            'economic_indicators': {},        # Dict of economic indicator values
            'market_regime': 'neutral',       # Current market regime: 'bull', 'bear', or 'neutral'
            'sector_performance': {},         # Recent sector performance data
            'current_cycle_stage': 'early_expansion',  # Current economic cycle stage
        }
        
        # Update parameters with defaults for any missing keys
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Strategy state
        self.sector_allocations = {}         # Current sector allocations
        self.target_allocations = {}         # Target sector allocations
        self.sector_signals = {}             # Current sector signals (buy/sell/hold)
        self.sector_rankings = {}            # Current sector rankings
        self.sector_data = {}                # Performance data for each sector
        self.current_cycle_stage = self.parameters['current_cycle_stage']  # Current economic cycle stage
        self.last_rebalance_date = None      # Date of last portfolio rebalance
        self.rebalance_step = 0              # Current step in gradual rotation
        self.defensive_mode = False          # Whether in defensive positioning
        self.sector_etf_positions = {}       # Current sector ETF positions
        self.sector_constituents = {}        # Stocks in each sector
        
        # Initialize sector data and portfolios
        self._initialize_sector_data()
        
        # Register for market events if event bus is available
        if self.event_bus:
            self.register_for_events(self.event_bus)
        
        logger.info(f"Initialized Sector Rotation Strategy with {self.parameters['rotation_model']} model")
    
    def register_for_events(self, event_bus: EventBus) -> None:
        """
        Register for relevant market events.
        
        Args:
            event_bus: EventBus to register with
        """
        # First register for common events via base class
        super().register_for_events(event_bus)
        
        # Register for sector-specific events
        event_bus.subscribe(EventType.MARKET_OPEN, self._on_market_open)
        event_bus.subscribe(EventType.ECONOMIC_DATA_RELEASE, self._on_economic_data_release)
        event_bus.subscribe(EventType.MARKET_REGIME_CHANGE, self._on_market_regime_change)
        event_bus.subscribe(EventType.REBALANCE_SIGNAL, self._on_rebalance_signal)
        
        logger.debug(f"Sector Rotation Strategy registered for events")
    
    def _initialize_sector_data(self) -> None:
        """
        Initialize sector data structures and default allocations.
        """
        # Initialize equal allocations across favored sectors for current cycle stage
        favored_sectors = self.ECONOMIC_CYCLE_SECTORS.get(self.current_cycle_stage, [])
        
        if not favored_sectors:
            # Default to equal weights across all sectors if no cycle stage defined
            all_sectors = list(self.SECTOR_ETFS.keys())
            num_sectors = min(len(all_sectors), self.parameters['max_sectors'])
            equal_weight = 1.0 / num_sectors
            
            self.sector_allocations = {sector: equal_weight for sector in all_sectors[:num_sectors]}
            self.target_allocations = copy.deepcopy(self.sector_allocations)
        else:
            # Allocate to favored sectors for current cycle stage
            num_sectors = min(len(favored_sectors), self.parameters['max_sectors'])
            equal_weight = 1.0 / num_sectors
            
            self.sector_allocations = {sector: equal_weight for sector in favored_sectors[:num_sectors]}
            self.target_allocations = copy.deepcopy(self.sector_allocations)
        
        # Initialize sector data with placeholder values
        for sector in self.SECTOR_ETFS:
            self.sector_data[sector] = {
                'performance_1m': 0.0,
                'performance_3m': 0.0,
                'performance_6m': 0.0,
                'relative_strength': 0.0,
                'momentum': 0.0,
                'volatility': 0.0,
                'cycle_score': 0.0,
                'composite_score': 0.0
            }
    
    def _on_market_open(self, event: Event) -> None:
        """
        Handle market open event.
        
        Check if rebalancing is needed based on frequency.
        
        Args:
            event: Market open event
        """
        current_date = datetime.now().date()
        
        # Check if we need to rebalance based on frequency
        if self._is_rebalance_due(current_date):
            logger.info(f"Rebalance due based on frequency: {self.parameters['rebalance_frequency']}")
            self._execute_rebalance()
    
    def _on_economic_data_release(self, event: Event) -> None:
        """
        Handle economic data release events.
        
        Update economic cycle stage forecast based on new data.
        
        Args:
            event: Economic data release event
        """
        if not self.parameters['enable_economic_forecasting']:
            return
        
        # Extract economic indicator data
        indicator_name = event.data.get('indicator_name')
        indicator_value = event.data.get('value')
        forecast_value = event.data.get('forecast')
        previous_value = event.data.get('previous')
        
        # Update our economic indicators
        if indicator_name:
            self.parameters['economic_indicators'][indicator_name] = {
                'value': indicator_value,
                'forecast': forecast_value,
                'previous': previous_value,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Updated economic indicator {indicator_name}: {indicator_value}")
            
            # Key indicators that might trigger cycle stage update
            key_indicators = [
                'gdp_growth', 'unemployment_rate', 'cpi', 'ppi', 
                'industrial_production', 'retail_sales', 'housing_starts'
            ]
            
            # If we received a key indicator, update cycle stage forecast
            if indicator_name in key_indicators:
                prev_stage = self.current_cycle_stage
                self._forecast_economic_cycle()
                
                if prev_stage != self.current_cycle_stage:
                    logger.info(f"Economic cycle stage changed from {prev_stage} to {self.current_cycle_stage}")
                    
                    # If cycle stage changed, consider rebalancing
                    if self.parameters['adapt_to_market_regime']:
                        self._update_sector_rankings()
                        self._calculate_target_allocations()
                        
                        # If changes are significant, trigger rebalance
                        if self._allocation_change_significant():
                            self._execute_rebalance()
    
    def _on_market_regime_change(self, event: Event) -> None:
        """
        Handle market regime change events.
        
        Adjust strategy parameters and potentially rebalance.
        
        Args:
            event: Market regime change event
        """
        if not self.parameters['adapt_to_market_regime']:
            return
            
        # Extract new market regime
        new_regime = event.data.get('regime')
        old_regime = self.parameters['market_regime']
        confidence = event.data.get('confidence', 0.5)
        
        if not new_regime or new_regime == old_regime:
            return
            
        logger.info(f"Market regime changed from {old_regime} to {new_regime} with confidence {confidence:.2f}")
        
        # Update our market regime
        self.parameters['market_regime'] = new_regime
        
        # Adjust strategy parameters based on new regime
        if new_regime == 'bear':
            # More defensive posture in bear markets
            if confidence > 0.7:  # High confidence bear market
                self.defensive_mode = True
                logger.info(f"Entering defensive mode due to high-confidence bear market")
                
                # Apply defensive allocation
                self.target_allocations = copy.deepcopy(self.parameters['defensive_allocation'])
                self._execute_rebalance()
            else:
                # Adjust weights but don't go fully defensive
                self.parameters['economic_cycle_weight'] = 0.5  # More weight on cycle
                self.parameters['momentum_weight'] = 0.2        # Less weight on momentum
                self._update_sector_rankings()
                self._calculate_target_allocations()
                self._execute_rebalance()
                
        elif new_regime == 'bull':
            # More aggressive posture in bull markets
            self.defensive_mode = False
            
            # Adjust weights to favor momentum in bull markets
            self.parameters['economic_cycle_weight'] = 0.3      # Less weight on cycle
            self.parameters['momentum_weight'] = 0.5            # More weight on momentum
            
            self._update_sector_rankings()
            self._calculate_target_allocations()
            self._execute_rebalance()
            
        else:  # neutral
            # Balanced approach in neutral markets
            self.defensive_mode = False
            
            # Reset to default weights
            self.parameters['economic_cycle_weight'] = 0.4
            self.parameters['momentum_weight'] = 0.3
            self.parameters['relative_strength_weight'] = 0.3
            
            self._update_sector_rankings()
            self._calculate_target_allocations()
            self._execute_rebalance()
    
    def _on_rebalance_signal(self, event: Event) -> None:
        """
        Handle explicit rebalance signal events.
        
        Args:
            event: Rebalance signal event
        """
        # Check if the rebalance signal is for our strategy
        strategy_name = event.data.get('strategy')
        if strategy_name and strategy_name != 'sector_rotation':
            return
            
        # Force rebalance
        logger.info(f"Received explicit rebalance signal for sector rotation strategy")
        self._update_sector_rankings()
        self._calculate_target_allocations()
        self._execute_rebalance()
    
    def _on_market_data_updated(self, event: Event) -> None:
        """
        Handle market data updated event.
        
        Update sector performance metrics.
        
        Args:
            event: Market data updated event
        """
        # Let the base class handle common functionality first
        super()._on_market_data_updated(event)
        
        # Check if this is data for a sector ETF
        symbol = event.data.get('symbol')
        sector = None
        
        # Find which sector this ETF corresponds to
        for sec, etf in self.SECTOR_ETFS.items():
            if etf == symbol:
                sector = sec
                break
        
        if not sector:
            return  # Not a sector ETF we're tracking
            
        # Update sector performance data if we have enough history
        if len(self.market_data) > 126:  # At least 6 months of daily data
            self._update_sector_performance(sector, symbol)
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed event.
        
        Potentially trigger rebalance on timeframe boundaries.
        
        Args:
            event: Timeframe completed event
        """
        # Let the base class handle common functionality first
        super()._on_timeframe_completed(event)
        
        # Check if this is the right timeframe for rebalancing decisions
        if self.session.timeframe == TimeFrame.DAY_1 and self.parameters['rebalance_frequency'] == 'daily':
            self._update_sector_rankings()
            self._calculate_target_allocations()
            
            # If allocations changed significantly, execute rebalance
            if self._allocation_change_significant():
                self._execute_rebalance()
                
        elif self.session.timeframe == TimeFrame.WEEK_1 and self.parameters['rebalance_frequency'] == 'weekly':
            self._update_sector_rankings()
            self._calculate_target_allocations()
            self._execute_rebalance()
    
    def _forecast_economic_cycle(self) -> str:
        """
        Forecast the current and upcoming economic cycle stage based on economic indicators.
        
        Returns:
            Forecasted economic cycle stage
        """
        if not self.parameters['enable_economic_forecasting']:
            return self.current_cycle_stage
            
        # Get economic indicators
        indicators = self.parameters['economic_indicators']
        
        # Simple scoring system for cycle stages
        cycle_scores = {
            'early_recession': 0,
            'late_recession': 0,
            'early_recovery': 0,
            'late_recovery': 0,
            'early_expansion': 0,
            'late_expansion': 0
        }
        
        # Analyze GDP growth (declining sharply = recession, strong growth = expansion)
        gdp_growth = indicators.get('gdp_growth', {}).get('value')
        if gdp_growth is not None:
            if gdp_growth < -1.0:  # Sharp decline
                cycle_scores['early_recession'] += 2
            elif gdp_growth < 0:   # Mild decline
                cycle_scores['late_expansion'] += 2
            elif gdp_growth < 1.0: # Weak growth
                cycle_scores['late_recession'] += 1
                cycle_scores['early_recovery'] += 1
            elif gdp_growth < 2.5: # Moderate growth
                cycle_scores['early_recovery'] += 2
                cycle_scores['late_recovery'] += 1
            elif gdp_growth < 4.0: # Strong growth
                cycle_scores['early_expansion'] += 2
            else:                  # Very strong growth
                cycle_scores['early_expansion'] += 1
                cycle_scores['late_recovery'] += 1
        
        # Analyze unemployment rate (rising = recession, falling = recovery/expansion)
        unemployment = indicators.get('unemployment_rate', {}).get('value')
        unemployment_prev = indicators.get('unemployment_rate', {}).get('previous')
        
        if unemployment is not None and unemployment_prev is not None:
            unemployment_change = unemployment - unemployment_prev
            
            if unemployment_change > 0.3:  # Sharp rise
                cycle_scores['early_recession'] += 2
            elif unemployment_change > 0:    # Mild rise
                cycle_scores['late_expansion'] += 1
                cycle_scores['early_recession'] += 1
            elif unemployment_change > -0.2: # Stable to slight decline
                cycle_scores['late_recession'] += 1
                cycle_scores['early_expansion'] += 1
            else:                            # Strong decline
                cycle_scores['early_recovery'] += 2
                cycle_scores['late_recovery'] += 1
        
        # Analyze inflation (CPI) (rising sharply = late expansion, falling = recession)
        cpi = indicators.get('cpi', {}).get('value')
        cpi_prev = indicators.get('cpi', {}).get('previous')
        
        if cpi is not None and cpi_prev is not None:
            cpi_change = cpi - cpi_prev
            
            if cpi_change > 1.0:    # Sharp rise
                cycle_scores['late_expansion'] += 3
            elif cpi_change > 0.5:  # Moderate rise
                cycle_scores['early_expansion'] += 2
                cycle_scores['late_expansion'] += 1
            elif cpi_change > 0:    # Mild rise
                cycle_scores['early_expansion'] += 1
                cycle_scores['late_recovery'] += 1
            elif cpi_change > -0.5: # Mild decline
                cycle_scores['early_recession'] += 1
            else:                   # Sharp decline
                cycle_scores['early_recession'] += 2
                cycle_scores['late_recession'] += 1
        
        # Analyze industrial production (rising = expansion, falling = recession)
        ind_prod = indicators.get('industrial_production', {}).get('value')
        ind_prod_prev = indicators.get('industrial_production', {}).get('previous')
        
        if ind_prod is not None and ind_prod_prev is not None:
            ind_prod_change = ind_prod - ind_prod_prev
            
            if ind_prod_change > 1.0:   # Strong rise
                cycle_scores['early_expansion'] += 2
                cycle_scores['late_recovery'] += 1
            elif ind_prod_change > 0.5: # Moderate rise
                cycle_scores['late_recovery'] += 2
                cycle_scores['early_expansion'] += 1
            elif ind_prod_change > 0:   # Mild rise
                cycle_scores['early_recovery'] += 2
            elif ind_prod_change > -0.5: # Mild decline
                cycle_scores['late_expansion'] += 1
                cycle_scores['early_recession'] += 1
            else:                        # Sharp decline
                cycle_scores['early_recession'] += 3
        
        # Find cycle stage with highest score
        forecasted_stage = max(cycle_scores.items(), key=lambda x: x[1])[0]
        
        # Only update if we have a strong signal or current stage is unknown
        if cycle_scores[forecasted_stage] >= 3 or self.current_cycle_stage == 'unknown':
            logger.info(f"Forecasted economic cycle stage: {forecasted_stage} (score: {cycle_scores[forecasted_stage]})")
            self.current_cycle_stage = forecasted_stage
            self.parameters['current_cycle_stage'] = forecasted_stage
        
        return self.current_cycle_stage
    
    def _update_sector_performance(self, sector: str, symbol: str) -> None:
        """
        Update performance metrics for a specific sector.
        
        Args:
            sector: Name of the sector
            symbol: ETF symbol for the sector
        """
        # Ensure we have data for this symbol
        if symbol not in self.market_data or len(self.market_data[symbol]) < 126:
            return
            
        sector_data = self.market_data[symbol]
        
        # Calculate performance metrics
        current_price = sector_data['close'].iloc[-1]
        
        # 1-month (21 trading days) performance
        if len(sector_data) > 21:
            month_ago_price = sector_data['close'].iloc[-22]  # -1 to get to the start of the period
            perf_1m = (current_price / month_ago_price - 1) * 100
        else:
            perf_1m = 0
        
        # 3-month (63 trading days) performance
        if len(sector_data) > 63:
            three_month_ago_price = sector_data['close'].iloc[-64]
            perf_3m = (current_price / three_month_ago_price - 1) * 100
        else:
            perf_3m = 0
        
        # 6-month (126 trading days) performance
        if len(sector_data) > 126:
            six_month_ago_price = sector_data['close'].iloc[-127]
            perf_6m = (current_price / six_month_ago_price - 1) * 100
        else:
            perf_6m = 0
        
        # Calculate momentum (rate of change of performance)
        if len(sector_data) > 30:  # At least 30 days needed
            # Price momentum over last 30 days
            prices_30d = sector_data['close'].iloc[-30:].values
            momentum = np.polyfit(np.arange(len(prices_30d)), prices_30d, 1)[0] / prices_30d.mean() * 100
        else:
            momentum = 0
        
        # Calculate volatility (standard deviation of daily returns)
        if len(sector_data) > 20:  # At least 20 days needed
            daily_returns = sector_data['close'].pct_change().iloc[-20:]
            volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized
        else:
            volatility = 0
        
        # Calculate relative strength (vs benchmark)
        if self.parameters['benchmark_index'] in self.market_data:
            benchmark_data = self.market_data[self.parameters['benchmark_index']]
            
            if len(benchmark_data) > 63 and len(sector_data) > 63:  # 3-month comparison
                sector_return = sector_data['close'].iloc[-1] / sector_data['close'].iloc[-64] - 1
                benchmark_return = benchmark_data['close'].iloc[-1] / benchmark_data['close'].iloc[-64] - 1
                
                relative_strength = (sector_return - benchmark_return) * 100
            else:
                relative_strength = 0
        else:
            relative_strength = 0
        
        # Update sector data
        self.sector_data[sector]['performance_1m'] = perf_1m
        self.sector_data[sector]['performance_3m'] = perf_3m
        self.sector_data[sector]['performance_6m'] = perf_6m
        self.sector_data[sector]['momentum'] = momentum
        self.sector_data[sector]['volatility'] = volatility
        self.sector_data[sector]['relative_strength'] = relative_strength
        
        # Also update in parameters for persistence
        if 'sector_performance' not in self.parameters:
            self.parameters['sector_performance'] = {}
        
        self.parameters['sector_performance'][sector] = {
            'performance_1m': perf_1m,
            'performance_3m': perf_3m,
            'performance_6m': perf_6m,
            'momentum': momentum,
            'volatility': volatility,
            'relative_strength': relative_strength,
            'last_updated': datetime.now().isoformat()
        }
    
    def _update_sector_rankings(self) -> None:
        """
        Update sector rankings based on the chosen rotation model.
        """
        rotation_model = self.parameters['rotation_model']
        
        # Update rankings based on the selected model
        if rotation_model == 'economic_cycle':
            self._rank_sectors_by_economic_cycle()
        elif rotation_model == 'momentum':
            self._rank_sectors_by_momentum()
        elif rotation_model == 'relative_strength':
            self._rank_sectors_by_relative_strength()
        else:  # combined
            self._rank_sectors_combined()
    
    def _rank_sectors_by_economic_cycle(self) -> None:
        """
        Rank sectors based on their expected performance in the current economic cycle stage.
        """
        # Ensure we have a valid cycle stage
        if not self.current_cycle_stage or self.current_cycle_stage not in self.ECONOMIC_CYCLE_SECTORS:
            # If no valid stage, forecast based on current data
            self._forecast_economic_cycle()
        
        # Get favored sectors for current cycle stage
        favored_sectors = self.ECONOMIC_CYCLE_SECTORS.get(self.current_cycle_stage, [])
        
        # Score sectors based on their position in the favored list
        scores = {}
        for sector in self.SECTOR_ETFS.keys():
            if sector in favored_sectors:
                # Higher score for sectors higher in the favored list
                scores[sector] = 10 - favored_sectors.index(sector)
            else:
                # Low base score for non-favored sectors
                scores[sector] = 1
        
        # Store scores in sector data
        for sector, score in scores.items():
            self.sector_data[sector]['cycle_score'] = score
        
        # Rank sectors by cycle score
        ranked_sectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        self.sector_rankings = {sector: i+1 for i, (sector, _) in enumerate(ranked_sectors)}
        
        logger.info(f"Ranked sectors by economic cycle ({self.current_cycle_stage}): " +
                   f"{', '.join([s for s, _ in ranked_sectors[:3]])}")
    
    def _rank_sectors_by_momentum(self) -> None:
        """
        Rank sectors based on price momentum.
        """
        # Calculate and rank momentum scores
        momentum_scores = {}
        
        for sector, data in self.sector_data.items():
            # Combined momentum score (more weight to recent momentum)
            momentum = data.get('momentum', 0)
            perf_1m = data.get('performance_1m', 0)
            perf_3m = data.get('performance_3m', 0)
            
            # Weighted momentum score
            momentum_score = (0.5 * momentum) + (0.3 * perf_1m) + (0.2 * perf_3m)
            momentum_scores[sector] = momentum_score
        
        # Rank sectors by momentum score
        ranked_sectors = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        self.sector_rankings = {sector: i+1 for i, (sector, _) in enumerate(ranked_sectors)}
        
        logger.info(f"Ranked sectors by momentum: {', '.join([s for s, _ in ranked_sectors[:3]])}")
    
    def _rank_sectors_by_relative_strength(self) -> None:
        """
        Rank sectors based on relative strength compared to the benchmark.
        """
        # Calculate and rank relative strength scores
        rs_scores = {}
        
        for sector, data in self.sector_data.items():
            # Combined relative strength score
            rs = data.get('relative_strength', 0)
            perf_3m = data.get('performance_3m', 0)
            perf_6m = data.get('performance_6m', 0)
            
            # Weighted relative strength score
            rs_score = (0.6 * rs) + (0.3 * perf_3m) + (0.1 * perf_6m)
            rs_scores[sector] = rs_score
        
        # Rank sectors by relative strength score
        ranked_sectors = sorted(rs_scores.items(), key=lambda x: x[1], reverse=True)
        self.sector_rankings = {sector: i+1 for i, (sector, _) in enumerate(ranked_sectors)}
        
        logger.info(f"Ranked sectors by relative strength: {', '.join([s for s, _ in ranked_sectors[:3]])}")
    
    def _rank_sectors_combined(self) -> None:
        """
        Rank sectors using a combined model of economic cycle, momentum, and relative strength.
        """
        # Get weights for each model
        cycle_weight = self.parameters['economic_cycle_weight']
        momentum_weight = self.parameters['momentum_weight']
        rs_weight = self.parameters['relative_strength_weight']
        
        # First calculate individual rankings
        self._rank_sectors_by_economic_cycle()
        cycle_rankings = self.sector_rankings.copy()
        
        self._rank_sectors_by_momentum()
        momentum_rankings = self.sector_rankings.copy()
        
        self._rank_sectors_by_relative_strength()
        rs_rankings = self.sector_rankings.copy()
        
        # Calculate combined scores (lower is better for rankings)
        combined_scores = {}
        
        for sector in self.SECTOR_ETFS.keys():
            cycle_rank = cycle_rankings.get(sector, len(self.SECTOR_ETFS))
            momentum_rank = momentum_rankings.get(sector, len(self.SECTOR_ETFS))
            rs_rank = rs_rankings.get(sector, len(self.SECTOR_ETFS))
            
            # Weighted rank score
            combined_score = (cycle_weight * cycle_rank) + (momentum_weight * momentum_rank) + (rs_weight * rs_rank)
            combined_scores[sector] = combined_score
            
            # Store composite score in sector data (normalize to 0-10 scale, 10 being best)
            max_possible_rank = len(self.SECTOR_ETFS)
            normalized_score = 10 - (combined_score / max_possible_rank * 10)
            self.sector_data[sector]['composite_score'] = normalized_score
        
        # Rank sectors by combined score (lower score = better rank)
        ranked_sectors = sorted(combined_scores.items(), key=lambda x: x[1])
        self.sector_rankings = {sector: i+1 for i, (sector, _) in enumerate(ranked_sectors)}
        
        logger.info(f"Ranked sectors by combined model: {', '.join([s for s, _ in ranked_sectors[:3]])}")
    
    def _calculate_target_allocations(self) -> None:
        """
        Calculate target sector allocations based on rankings and allocation parameters.
        """
        # Handle defensive mode differently
        if self.defensive_mode:
            self.target_allocations = copy.deepcopy(self.parameters['defensive_allocation'])
            return
        
        # Get maximum number of sectors to include
        max_sectors = self.parameters['max_sectors']
        
        # Get concentration method
        concentration = self.parameters['sector_concentration']
        
        # Sort sectors by ranking
        sorted_sectors = sorted(self.sector_rankings.items(), key=lambda x: x[1])[:max_sectors]
        selected_sectors = [sector for sector, _ in sorted_sectors]
        
        # Calculate allocations based on concentration method
        if concentration == 'equal_weight':
            weight = 1.0 / len(selected_sectors)
            allocations = {sector: weight for sector in selected_sectors}
            
        elif concentration == 'market_weight':
            # In real implementation, this would use actual market cap weights
            # Here using simplified weights based on ranking
            total_weight = sum(range(1, len(selected_sectors) + 1))
            allocations = {}
            
            for i, sector in enumerate(selected_sectors):
                # Higher weight to higher-ranked sectors
                allocations[sector] = (len(selected_sectors) - i) / total_weight
                
        elif concentration == 'custom':
            # Use custom weights if provided
            custom_weights = self.parameters.get('custom_sector_weights', {})
            
            # Filter to selected sectors and normalize
            filtered_weights = {s: w for s, w in custom_weights.items() if s in selected_sectors}
            total_weight = sum(filtered_weights.values())
            
            if total_weight > 0:
                allocations = {s: w / total_weight for s, w in filtered_weights.items()}
            else:
                # Fallback to equal weight if custom weights not available or sum to zero
                weight = 1.0 / len(selected_sectors)
                allocations = {sector: weight for sector in selected_sectors}
        else:
            # Default to equal weight
            weight = 1.0 / len(selected_sectors)
            allocations = {sector: weight for sector in selected_sectors}
        
        # Apply minimum and maximum allocation constraints
        min_allocation = self.parameters['min_sector_allocation']
        max_allocation = self.parameters['max_sector_allocation']
        
        # First pass: cap at maximum
        for sector in allocations:
            if allocations[sector] > max_allocation:
                allocations[sector] = max_allocation
                
        # Redistribute excess weight
        total_allocation = sum(allocations.values())
        if total_allocation < 1.0:
            # Distribute remaining weight proportionally
            remaining = 1.0 - total_allocation
            
            uncapped_sectors = [s for s in allocations if allocations[s] < max_allocation]
            if uncapped_sectors:
                total_uncapped = sum(allocations[s] for s in uncapped_sectors)
                
                for sector in uncapped_sectors:
                    if total_uncapped > 0:
                        allocations[sector] += remaining * (allocations[sector] / total_uncapped)
        
        # Second pass: ensure minimum allocation
        for sector in allocations:
            if allocations[sector] < min_allocation:
                allocations[sector] = min_allocation
                
        # Final normalization to ensure sum is 1.0
        total_allocation = sum(allocations.values())
        if total_allocation != 1.0:
            for sector in allocations:
                allocations[sector] /= total_allocation
        
        # Update target allocations
        self.target_allocations = allocations
        
        # Log the new target allocations
        allocation_str = ", ".join([f"{s}: {allocations[s]:.2f}" for s in allocations])
        logger.info(f"Calculated target allocations: {allocation_str}")
    
    def _is_rebalance_due(self, current_date: datetime.date) -> bool:
        """
        Check if a rebalance is due based on frequency and last rebalance date.
        
        Args:
            current_date: Current date to check against
            
        Returns:
            True if rebalance is due, False otherwise
        """
        # If never rebalanced, it's due
        if self.last_rebalance_date is None:
            return True
            
        # Get days since last rebalance
        days_since_rebalance = (current_date - self.last_rebalance_date).days
        
        # Check based on frequency
        frequency = self.parameters['rebalance_frequency']
        
        if frequency == 'daily':
            return days_since_rebalance >= 1
        elif frequency == 'weekly':
            return days_since_rebalance >= 7
        elif frequency == 'monthly':
            return days_since_rebalance >= 30
        elif frequency == 'quarterly':
            return days_since_rebalance >= 90
        else:
            # Default to monthly
            return days_since_rebalance >= 30
    
    def _allocation_change_significant(self) -> bool:
        """
        Check if the change in allocations is significant enough to warrant a rebalance.
        
        Returns:
            True if changes are significant, False otherwise
        """
        # Get the threshold for minimum outperformance
        threshold = self.parameters['minimum_outperformance'] / 100.0  # Convert from percentage
        
        # Compare current and target allocations
        for sector, target in self.target_allocations.items():
            current = self.sector_allocations.get(sector, 0.0)
            
            # If any sector allocation change exceeds threshold, consider it significant
            if abs(target - current) > threshold:
                logger.info(f"Significant allocation change detected for {sector}: {current:.2f} to {target:.2f}")
                return True
                
        # Check for new sectors in target that aren't in current
        new_sectors = [s for s in self.target_allocations if s not in self.sector_allocations]
        if new_sectors and any(self.target_allocations[s] > threshold for s in new_sectors):
            logger.info(f"New sectors added to allocation: {', '.join(new_sectors)}")
            return True
            
        # Check for removed sectors
        removed_sectors = [s for s in self.sector_allocations if s not in self.target_allocations]
        if removed_sectors and any(self.sector_allocations[s] > threshold for s in removed_sectors):
            logger.info(f"Sectors removed from allocation: {', '.join(removed_sectors)}")
            return True
        
        logger.debug("Allocation changes are not significant enough for rebalance")
        return False
    
    def _execute_rebalance(self) -> None:
        """
        Execute portfolio rebalance to align with target allocations.
        """
        # Skip if no target allocations
        if not self.target_allocations:
            logger.warning("No target allocations calculated, skipping rebalance")
            return
            
        # Check if we want to do gradual rotation
        if self.parameters['gradual_rotation'] and self.rebalance_step < self.parameters['rotation_steps']:
            # Calculate intermediate target based on step
            self._execute_gradual_rebalance()
        else:
            # Execute full rebalance
            self._execute_full_rebalance()
            
        # Update last rebalance date
        self.last_rebalance_date = datetime.now().date()
        
        # Reset gradual rebalance step if we've completed the rotation
        if self.rebalance_step >= self.parameters['rotation_steps']:
            self.rebalance_step = 0
            
        # Log rebalance completion
        logger.info(f"Completed portfolio rebalance (step {self.rebalance_step}/{self.parameters['rotation_steps']})")
    
    def _execute_gradual_rebalance(self) -> None:
        """
        Execute a gradual rebalance toward target allocations.
        """
        # Calculate intermediate target based on step
        step = self.rebalance_step + 1
        total_steps = self.parameters['rotation_steps']
        
        # Calculate intermediate allocations
        intermediate_allocations = {}
        
        # For each sector in either current or target
        all_sectors = set(list(self.sector_allocations.keys()) + list(self.target_allocations.keys()))
        
        for sector in all_sectors:
            current = self.sector_allocations.get(sector, 0.0)
            target = self.target_allocations.get(sector, 0.0)
            
            # Linear interpolation between current and target
            intermediate = current + (target - current) * (step / total_steps)
            
            # Only include sectors with meaningful allocation
            if intermediate > 0.01:  # 1% minimum to be included
                intermediate_allocations[sector] = intermediate
            
        # Normalize to ensure sum is 1.0
        total_allocation = sum(intermediate_allocations.values())
        if total_allocation != 0:
            for sector in intermediate_allocations:
                intermediate_allocations[sector] /= total_allocation
        
        # Generate signals for the intermediate allocation
        self._generate_rebalance_signals(intermediate_allocations)
        
        # Update current allocations
        self.sector_allocations = intermediate_allocations
        
        # Increment step
        self.rebalance_step += 1
        
        # Log intermediate allocations
        allocation_str = ", ".join([f"{s}: {intermediate_allocations[s]:.2f}" for s in intermediate_allocations])
        logger.info(f"Gradual rebalance step {step}/{total_steps}: {allocation_str}")
    
    def _execute_full_rebalance(self) -> None:
        """
        Execute a full rebalance to target allocations.
        """
        # Generate signals for the target allocation
        self._generate_rebalance_signals(self.target_allocations)
        
        # Update current allocations to match target
        self.sector_allocations = copy.deepcopy(self.target_allocations)
        
        # Log full rebalance
        allocation_str = ", ".join([f"{s}: {self.target_allocations[s]:.2f}" for s in self.target_allocations])
        logger.info(f"Full rebalance completed: {allocation_str}")
    
    def _generate_rebalance_signals(self, target_allocations: Dict[str, float]) -> None:
        """
        Generate buy/sell signals to achieve the target allocation.
        
        Args:
            target_allocations: Target sector allocations
        """
        # Calculate differences between current positions and target allocations
        buys = {}
        sells = {}
        holds = {}
        
        # Assume we're trading sector ETFs
        if self.parameters['trade_sector_etfs']:
            # For each sector, determine if we need to buy, sell, or hold
            for sector, target_weight in target_allocations.items():
                # Get ETF symbol for this sector
                etf_symbol = self.SECTOR_ETFS.get(sector)
                if not etf_symbol:
                    continue
                
                # Get current position weight if any
                current_weight = 0.0
                for position in self.positions:
                    if position.symbol == etf_symbol and position.status == PositionStatus.OPEN:
                        # Calculate actual weight based on current market value
                        if 'current_value' in position.metadata:
                            current_weight = position.metadata['current_value'] / self._get_total_portfolio_value()
                        else:
                            # Fallback to allocation if value not available
                            current_weight = self.sector_allocations.get(sector, 0.0)
                
                # Calculate the difference
                diff = target_weight - current_weight
                
                # Apply a small threshold to avoid tiny rebalance trades
                threshold = 0.005  # 0.5% minimum change to trigger a trade
                
                if diff > threshold:
                    buys[sector] = (etf_symbol, diff)
                elif diff < -threshold:
                    sells[sector] = (etf_symbol, -diff)
                else:
                    holds[sector] = (etf_symbol, current_weight)
            
            # Generate signals for sells first (to free up capital)
            for sector, (symbol, weight) in sells.items():
                self._generate_sell_signal(symbol, weight, reason=f"Sector rebalance: reduce {sector}")
            
            # Then generate signals for buys
            for sector, (symbol, weight) in buys.items():
                self._generate_buy_signal(symbol, weight, reason=f"Sector rebalance: increase {sector}")
        else:
            # If not trading ETFs, we'd need to handle individual stocks here
            # This would involve selecting stocks within each sector and generating signals for those
            logger.info("Rebalancing with individual stocks is not implemented yet")
        
        # Store current sector signals
        self.sector_signals = {
            'buys': buys,
            'sells': sells,
            'holds': holds,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_buy_signal(self, symbol: str, weight: float, reason: str = "") -> None:
        """
        Generate a buy signal for a specific symbol.
        
        Args:
            symbol: Symbol to buy
            weight: Target portfolio weight
            reason: Reason for the signal
        """
        # Create a unique signal ID
        signal_id = str(uuid.uuid4())
        
        # Get current price if available in market data
        current_price = None
        if symbol in self.market_data and len(self.market_data[symbol]) > 0:
            current_price = self.market_data[symbol]['close'].iloc[-1]
        else:
            logger.warning(f"No market data available for {symbol}, cannot generate buy signal")
            return
        
        # Create signal object
        signal = Signal(
            id=signal_id,
            symbol=symbol,
            signal_type=SignalType.LONG,
            entry_price=current_price,
            stop_loss=None,  # Sector ETFs don't typically use stop losses
            target_price=None,
            timestamp=datetime.now(),
            expiration=datetime.now() + timedelta(days=1),
            confidence=0.8,  # High confidence for rebalance trades
            metadata={
                'strategy': 'sector_rotation',
                'portfolio_weight': weight,
                'reason': reason,
                'economic_cycle_stage': self.current_cycle_stage
            }
        )
        
        # Log signal generation
        logger.info(f"Generated BUY signal for {symbol} with weight {weight:.2f} ({reason})")
        
        # Act on the signal
        self._act_on_signal(signal)
    
    def _generate_sell_signal(self, symbol: str, weight: float, reason: str = "") -> None:
        """
        Generate a sell signal for a specific symbol.
        
        Args:
            symbol: Symbol to sell
            weight: Portfolio weight to reduce
            reason: Reason for the signal
        """
        # Find existing position for this symbol
        position_id = None
        for position in self.positions:
            if position.symbol == symbol and position.status == PositionStatus.OPEN:
                position_id = position.id
                break
        
        if not position_id:
            logger.warning(f"No open position found for {symbol}, cannot generate sell signal")
            return
        
        # Close the position with the specified reason
        self._close_position(position_id, reason)
    
    def _act_on_signal(self, signal: Signal) -> None:
        """
        Act on a generated signal.
        
        Args:
            signal: Signal object with trade details
        """
        # Check if we already have a position for this symbol
        existing_position = None
        for position in self.positions:
            if position.symbol == signal.symbol and position.status == PositionStatus.OPEN:
                existing_position = position
                break
        
        # If we already have a position, update its target weight
        if existing_position:
            existing_position.metadata['target_weight'] = signal.metadata.get('portfolio_weight', 0.0)
            logger.info(f"Updated target weight for existing position {signal.symbol}: {existing_position.metadata['target_weight']:.2f}")
            return
        
        # Calculate position size based on portfolio weight
        portfolio_value = self._get_total_portfolio_value()
        target_value = portfolio_value * signal.metadata.get('portfolio_weight', 0.0)
        
        if signal.entry_price and signal.entry_price > 0:
            position_size = target_value / signal.entry_price
        else:
            logger.warning(f"Invalid entry price for {signal.symbol}, cannot calculate position size")
            return
        
        # Round to whole number of shares
        position_size = round(position_size)
        
        # Create a new position
        position_id = str(uuid.uuid4())
        position = Position(
            id=position_id,
            symbol=signal.symbol,
            direction='long',  # All sector positions are long
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            target_price=signal.target_price,
            size=position_size,
            entry_time=datetime.now(),
            status=PositionStatus.PENDING,
            metadata={
                'strategy': 'sector_rotation',
                'signal_id': signal.id,
                'target_weight': signal.metadata.get('portfolio_weight', 0.0),
                'economic_cycle_stage': self.current_cycle_stage,
                'reason': signal.metadata.get('reason', '')
            }
        )
        
        # In a real implementation, this would send the order to a broker
        # and the position would be updated once the order is filled
        logger.info(f"Opening position for {position.symbol} " +
                   f"at {position.entry_price:.2f} with size {position.size:.0f} shares " +
                   f"(weight: {signal.metadata.get('portfolio_weight', 0.0):.2f})")
        
        # Add position to our tracker
        self.positions.append(position)
        
        # Update position status to OPEN (in real implementation this would happen after fill)
        position.status = PositionStatus.OPEN
        
        # Track the sector ETF position
        for sector, symbol in self.SECTOR_ETFS.items():
            if symbol == position.symbol:
                self.sector_etf_positions[sector] = position.id
                break
        
        # Emit an event for position opened if event bus is available
        if self.event_bus:
            event = Event(
                event_type=EventType.POSITION_OPENED,
                timestamp=datetime.now(),
                data={
                    'position_id': position.id,
                    'symbol': position.symbol,
                    'direction': position.direction,
                    'entry_price': position.entry_price,
                    'size': position.size,
                    'strategy': 'sector_rotation',
                    'target_weight': signal.metadata.get('portfolio_weight', 0.0)
                }
            )
            self.event_bus.emit(event)
    
    def _close_position(self, position_id: str, reason: str = "Unspecified") -> None:
        """
        Close an open position.
        
        Args:
            position_id: ID of the position to close
            reason: Reason for closing the position
        """
        # Find the position in our tracker
        position = None
        for p in self.positions:
            if p.id == position_id:
                position = p
                break
        
        if position is None or position.status != PositionStatus.OPEN:
            logger.warning(f"Cannot close position {position_id}: not found or not open")
            return
        
        # Get current price (simulated - in real implementation would get from market)
        current_price = None
        if position.symbol in self.market_data and len(self.market_data[position.symbol]) > 0:
            current_price = self.market_data[position.symbol]['close'].iloc[-1]
        else:
            # Fallback to entry price if no current price available
            current_price = position.entry_price
        
        # Calculate P&L
        pnl = (current_price - position.entry_price) * position.size
        
        # Update position status
        position.status = PositionStatus.CLOSED
        position.exit_price = current_price
        position.exit_time = datetime.now()
        position.pnl = pnl
        position.metadata['exit_reason'] = reason
        
        logger.info(f"Closed position for {position.symbol} " +
                  f"at {position.exit_price:.2f} with P&L {position.pnl:.2f} " +
                  f"(reason: {reason})")
        
        # Remove from sector ETF positions tracker
        for sector, pos_id in list(self.sector_etf_positions.items()):
            if pos_id == position_id:
                del self.sector_etf_positions[sector]
                break
        
        # Emit an event for position closed if event bus is available
        if self.event_bus:
            event = Event(
                event_type=EventType.POSITION_CLOSED,
                timestamp=datetime.now(),
                data={
                    'position_id': position.id,
                    'symbol': position.symbol,
                    'direction': position.direction,
                    'entry_price': position.entry_price,
                    'exit_price': position.exit_price,
                    'pnl': position.pnl,
                    'reason': reason,
                    'strategy': 'sector_rotation'
                }
            )
            self.event_bus.emit(event)
    
    def _get_total_portfolio_value(self) -> float:
        """
        Calculate total portfolio value based on current positions.
        
        Returns:
            Total portfolio value
        """
        # In a real implementation, this would get the actual account value
        # For simulation, using a base value of $100,000
        base_value = 100000.0
        
        # Add value of current positions
        position_value = 0.0
        for position in self.positions:
            if position.status == PositionStatus.OPEN:
                # Get current price if available
                current_price = position.entry_price  # Fallback
                if position.symbol in self.market_data and len(self.market_data[position.symbol]) > 0:
                    current_price = self.market_data[position.symbol]['close'].iloc[-1]
                
                # Calculate position value
                position_value += current_price * position.size
                
                # Update position metadata with current value
                position.metadata['current_value'] = current_price * position.size
        
        return base_value
    
    def get_current_allocations(self) -> Dict[str, Any]:
        """
        Get current sector allocations and portfolio status.
        
        Returns:
            Dictionary with current allocations and portfolio status
        """
        result = {
            'current_allocations': copy.deepcopy(self.sector_allocations),
            'target_allocations': copy.deepcopy(self.target_allocations),
            'economic_cycle_stage': self.current_cycle_stage,
            'defensive_mode': self.defensive_mode,
            'last_rebalance_date': self.last_rebalance_date.isoformat() if self.last_rebalance_date else None,
            'sector_rankings': copy.deepcopy(self.sector_rankings),
            'rotation_model': self.parameters['rotation_model'],
            'positions': []
        }
        
        # Add position information
        for position in self.positions:
            if position.status == PositionStatus.OPEN:
                result['positions'].append({
                    'symbol': position.symbol,
                    'size': position.size,
                    'entry_price': position.entry_price,
                    'current_value': position.metadata.get('current_value', position.size * position.entry_price),
                    'weight': position.metadata.get('target_weight', 0.0),
                    'sector': next((sect for sect, sym in self.SECTOR_ETFS.items() if sym == position.symbol), None)
                })
        
        return result
