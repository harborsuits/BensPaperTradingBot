#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Risk Factors

This module implements advanced risk factor calculations for the risk management engine,
including liquidity risk, factor tilts, sector exposure, and other sophisticated
risk analytics.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

class EnhancedRiskFactors:
    """
    Calculates and monitors advanced risk factors for more sophisticated
    risk management and strategy rotation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced risk factors calculator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Initialize factor exposure tracking
        self.factor_exposures = {
            'value': 0.0,
            'momentum': 0.0,
            'size': 0.0,
            'quality': 0.0,
            'volatility': 0.0,
            'growth': 0.0,
            'yield': 0.0,
            'liquidity': 0.0
        }
        
        # Initialize sector exposure tracking
        self.sector_exposures = defaultdict(float)
        
        # Initialize geographic exposure tracking
        self.geographic_exposures = defaultdict(float)
        
        # Current factor limits
        self.factor_limits = self.config.get('factor_exposure_limits', {
            'value': 0.4,
            'momentum': 0.4,
            'size': 0.3,
            'quality': 0.4,
            'volatility': 0.3,
            'growth': 0.3,
            'yield': 0.3,
            'liquidity': 0.3
        })
        
        # Sector concentration limits
        self.sector_limits = self.config.get('sector_concentration_limits', {
            'default': 0.3  # Default 30% max per sector
        })
        
        logger.info("Enhanced risk factors module initialized")
    
    def calculate_liquidity_risk(self, positions: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Calculate liquidity risk for current positions based on position size,
        average daily volume, and bid-ask spread.
        
        Args:
            positions: Dictionary of positions with market data
            
        Returns:
            Liquidity risk assessment
        """
        total_portfolio_value = sum(
            pos.get('market_value', 0) 
            for pos in positions.values()
        )
        
        if total_portfolio_value == 0:
            return {
                "portfolio_liquidity_score": 1.0,
                "days_to_liquidate": 0,
                "liquidity_risk_level": "low",
                "positions_at_risk": [],
                "timestamp": datetime.now().isoformat()
            }
        
        liquidity_scores = {}
        days_to_liquidate = {}
        positions_at_risk = []
        
        for symbol, position in positions.items():
            # Get position details
            position_size = position.get('quantity', 0)
            market_value = position.get('market_value', 0)
            avg_daily_volume = position.get('avg_daily_volume', 0)
            bid_ask_spread = position.get('bid_ask_spread', 0.01)  # Default 1%
            
            # Skip if essential data is missing
            if not all([position_size, market_value, avg_daily_volume]):
                continue
            
            # Calculate position as percentage of ADV
            pct_of_adv = abs(position_size) / avg_daily_volume if avg_daily_volume > 0 else 1.0
            
            # Calculate days to liquidate (assuming 10% of ADV can be traded without impact)
            days_to_liquidate[symbol] = pct_of_adv / 0.1
            
            # Calculate liquidity score (0-1, higher is more liquid)
            # Based on days to liquidate and bid-ask spread
            liquidity_score = 1.0 / (1.0 + days_to_liquidate[symbol])
            
            # Penalize for wide bid-ask spread
            liquidity_score *= (1.0 - min(bid_ask_spread, 0.2) / 0.2)
            
            liquidity_scores[symbol] = max(0.01, min(1.0, liquidity_score))
            
            # Determine if position is at liquidity risk
            if liquidity_scores[symbol] < 0.4:  # Low liquidity threshold
                positions_at_risk.append({
                    "symbol": symbol,
                    "liquidity_score": liquidity_scores[symbol],
                    "days_to_liquidate": days_to_liquidate[symbol],
                    "pct_of_adv": pct_of_adv,
                    "market_value": market_value,
                    "pct_of_portfolio": market_value / total_portfolio_value
                })
        
        # Calculate portfolio-level liquidity score (weighted by position value)
        portfolio_liquidity_score = 0.0
        for symbol, position in positions.items():
            if symbol in liquidity_scores:
                market_value = position.get('market_value', 0)
                weight = market_value / total_portfolio_value
                portfolio_liquidity_score += liquidity_scores[symbol] * weight
        
        # Calculate overall days to liquidate (weighted)
        portfolio_days_to_liquidate = 0.0
        for symbol, position in positions.items():
            if symbol in days_to_liquidate:
                market_value = position.get('market_value', 0)
                weight = market_value / total_portfolio_value
                portfolio_days_to_liquidate += days_to_liquidate[symbol] * weight
        
        # Determine liquidity risk level
        liquidity_risk_level = "low"
        if portfolio_liquidity_score < 0.4:
            liquidity_risk_level = "high"
        elif portfolio_liquidity_score < 0.7:
            liquidity_risk_level = "medium"
        
        return {
            "portfolio_liquidity_score": portfolio_liquidity_score,
            "days_to_liquidate": portfolio_days_to_liquidate,
            "liquidity_risk_level": liquidity_risk_level,
            "positions_at_risk": positions_at_risk,
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_factor_exposures(
            self, 
            positions: Dict[str, Dict],
            factor_data: Dict[str, Dict]
        ) -> Dict[str, Any]:
        """
        Calculate exposure to common risk factors like value, momentum, size, etc.
        
        Args:
            positions: Dictionary of positions
            factor_data: Dictionary of factor exposures by symbol
            
        Returns:
            Factor exposure assessment
        """
        total_portfolio_value = sum(
            pos.get('market_value', 0) 
            for pos in positions.values()
        )
        
        if total_portfolio_value == 0 or not factor_data:
            return {
                "factor_exposures": self.factor_exposures.copy(),
                "factor_risk_level": "low",
                "factors_at_risk": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Reset factor exposures
        for factor in self.factor_exposures:
            self.factor_exposures[factor] = 0.0
        
        # Calculate weighted factor exposures
        for symbol, position in positions.items():
            market_value = position.get('market_value', 0)
            weight = market_value / total_portfolio_value
            
            # Get factor data for this symbol
            symbol_factors = factor_data.get(symbol, {})
            
            # Accumulate weighted factor exposures
            for factor, exposure in symbol_factors.items():
                if factor in self.factor_exposures:
                    self.factor_exposures[factor] += exposure * weight
        
        # Identify factors exceeding limits
        factors_at_risk = []
        for factor, exposure in self.factor_exposures.items():
            limit = self.factor_limits.get(factor, 0.4)
            if abs(exposure) > limit:
                factors_at_risk.append({
                    "factor": factor,
                    "exposure": exposure,
                    "limit": limit,
                    "excess": abs(exposure) - limit
                })
        
        # Determine overall factor risk level
        factor_risk_level = "low"
        if len(factors_at_risk) > 2:
            factor_risk_level = "high"
        elif len(factors_at_risk) > 0:
            factor_risk_level = "medium"
        
        return {
            "factor_exposures": self.factor_exposures.copy(),
            "factor_risk_level": factor_risk_level,
            "factors_at_risk": factors_at_risk,
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_sector_exposures(
            self, 
            positions: Dict[str, Dict],
            sector_data: Dict[str, str]
        ) -> Dict[str, Any]:
        """
        Calculate sector concentration risk.
        
        Args:
            positions: Dictionary of positions
            sector_data: Dictionary mapping symbols to sectors
            
        Returns:
            Sector exposure assessment
        """
        total_portfolio_value = sum(
            pos.get('market_value', 0) 
            for pos in positions.values()
        )
        
        if total_portfolio_value == 0 or not sector_data:
            return {
                "sector_exposures": dict(self.sector_exposures),
                "sector_risk_level": "low",
                "sectors_at_risk": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Reset sector exposures
        self.sector_exposures = defaultdict(float)
        
        # Calculate sector exposures
        for symbol, position in positions.items():
            market_value = position.get('market_value', 0)
            weight = market_value / total_portfolio_value
            
            # Get sector for this symbol
            sector = sector_data.get(symbol, "Unknown")
            self.sector_exposures[sector] += weight
        
        # Identify sectors exceeding concentration limits
        sectors_at_risk = []
        for sector, exposure in self.sector_exposures.items():
            limit = self.sector_limits.get(sector, self.sector_limits.get('default', 0.3))
            if exposure > limit:
                sectors_at_risk.append({
                    "sector": sector,
                    "exposure": exposure,
                    "limit": limit,
                    "excess": exposure - limit
                })
        
        # Determine overall sector risk level
        sector_risk_level = "low"
        if sum(exposure for sector, exposure in sectors_at_risk) > 0.5:
            sector_risk_level = "high"
        elif len(sectors_at_risk) > 0:
            sector_risk_level = "medium"
        
        return {
            "sector_exposures": dict(self.sector_exposures),
            "sector_risk_level": sector_risk_level,
            "sectors_at_risk": sectors_at_risk,
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_geographic_exposures(
            self, 
            positions: Dict[str, Dict],
            country_data: Dict[str, str]
        ) -> Dict[str, Any]:
        """
        Calculate geographic concentration risk.
        
        Args:
            positions: Dictionary of positions
            country_data: Dictionary mapping symbols to countries
            
        Returns:
            Geographic exposure assessment
        """
        total_portfolio_value = sum(
            pos.get('market_value', 0) 
            for pos in positions.values()
        )
        
        if total_portfolio_value == 0 or not country_data:
            return {
                "geographic_exposures": dict(self.geographic_exposures),
                "geographic_risk_level": "low",
                "regions_at_risk": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Reset geographic exposures
        self.geographic_exposures = defaultdict(float)
        
        # Calculate geographic exposures
        for symbol, position in positions.items():
            market_value = position.get('market_value', 0)
            weight = market_value / total_portfolio_value
            
            # Get country for this symbol
            country = country_data.get(symbol, "Unknown")
            self.geographic_exposures[country] += weight
        
        # Group countries into regions
        region_exposures = defaultdict(float)
        country_to_region = {
            "USA": "North America",
            "Canada": "North America",
            "Mexico": "North America",
            "UK": "Europe",
            "Germany": "Europe",
            "France": "Europe",
            "Italy": "Europe",
            "Spain": "Europe",
            "China": "Asia",
            "Japan": "Asia",
            "India": "Asia",
            "Australia": "Asia-Pacific",
            "Brazil": "Latin America",
            "Russia": "Emerging Markets",
            # Add more mappings as needed
        }
        
        for country, exposure in self.geographic_exposures.items():
            region = country_to_region.get(country, "Other")
            region_exposures[region] += exposure
        
        # Identify regions exceeding concentration limits
        region_limits = {
            "North America": 0.7,
            "Europe": 0.4,
            "Asia": 0.4,
            "Asia-Pacific": 0.3,
            "Latin America": 0.2,
            "Emerging Markets": 0.3,
            "Other": 0.2
        }
        
        regions_at_risk = []
        for region, exposure in region_exposures.items():
            limit = region_limits.get(region, 0.3)
            if exposure > limit:
                regions_at_risk.append({
                    "region": region,
                    "exposure": exposure,
                    "limit": limit,
                    "excess": exposure - limit
                })
        
        # Determine overall geographic risk level
        geographic_risk_level = "low"
        if sum(exposure for region, exposure in regions_at_risk) > 0.5:
            geographic_risk_level = "high"
        elif len(regions_at_risk) > 0:
            geographic_risk_level = "medium"
        
        return {
            "geographic_exposures": dict(self.geographic_exposures),
            "region_exposures": dict(region_exposures),
            "geographic_risk_level": geographic_risk_level,
            "regions_at_risk": regions_at_risk,
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_concentration_risk(self, positions: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Calculate position concentration risk.
        
        Args:
            positions: Dictionary of positions
            
        Returns:
            Concentration risk assessment
        """
        total_portfolio_value = sum(
            pos.get('market_value', 0) 
            for pos in positions.values()
        )
        
        if total_portfolio_value == 0:
            return {
                "concentration_score": 0.0,
                "herfindahl_index": 0.0,
                "top_positions": [],
                "concentration_risk_level": "low",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate position weights
        weights = {}
        for symbol, position in positions.items():
            market_value = position.get('market_value', 0)
            weights[symbol] = market_value / total_portfolio_value
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        # This is a measure of concentration (sum of squared weights)
        hhi = sum(weight ** 2 for weight in weights.values())
        
        # HHI ranges from 1/N (perfectly diversified) to 1 (single position)
        # Normalize to 0-1 range
        n = len(weights)
        min_hhi = 1 / n if n > 0 else 0
        normalized_hhi = (hhi - min_hhi) / (1 - min_hhi) if n > 1 else 1
        
        # Identify top positions
        top_positions = sorted(
            [{"symbol": symbol, "weight": weight} for symbol, weight in weights.items()],
            key=lambda x: x["weight"],
            reverse=True
        )[:5]  # Top 5 positions
        
        # Calculate top 3 concentration
        top3_concentration = sum(pos["weight"] for pos in top_positions[:3])
        
        # Determine concentration risk level
        concentration_risk_level = "low"
        if normalized_hhi > 0.7 or top3_concentration > 0.5:
            concentration_risk_level = "high"
        elif normalized_hhi > 0.4 or top3_concentration > 0.3:
            concentration_risk_level = "medium"
        
        return {
            "concentration_score": normalized_hhi,
            "herfindahl_index": hhi,
            "top3_concentration": top3_concentration,
            "top_positions": top_positions,
            "concentration_risk_level": concentration_risk_level,
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_all_risk_factors(
            self,
            positions: Dict[str, Dict],
            factor_data: Optional[Dict[str, Dict]] = None,
            sector_data: Optional[Dict[str, str]] = None,
            country_data: Optional[Dict[str, str]] = None
        ) -> Dict[str, Any]:
        """
        Calculate all enhanced risk factors.
        
        Args:
            positions: Dictionary of positions
            factor_data: Optional dictionary of factor exposures by symbol
            sector_data: Optional dictionary mapping symbols to sectors
            country_data: Optional dictionary mapping symbols to countries
            
        Returns:
            Comprehensive risk factor assessment
        """
        results = {}
        
        # Calculate liquidity risk
        results["liquidity_risk"] = self.calculate_liquidity_risk(positions)
        
        # Calculate factor exposures if data available
        if factor_data:
            results["factor_risk"] = self.calculate_factor_exposures(positions, factor_data)
        
        # Calculate sector exposures if data available
        if sector_data:
            results["sector_risk"] = self.calculate_sector_exposures(positions, sector_data)
        
        # Calculate geographic exposures if data available
        if country_data:
            results["geographic_risk"] = self.calculate_geographic_exposures(positions, country_data)
        
        # Calculate concentration risk
        results["concentration_risk"] = self.calculate_concentration_risk(positions)
        
        # Determine overall enhanced risk level
        risk_levels = [
            results.get("liquidity_risk", {}).get("liquidity_risk_level", "low"),
            results.get("factor_risk", {}).get("factor_risk_level", "low"),
            results.get("sector_risk", {}).get("sector_risk_level", "low"),
            results.get("geographic_risk", {}).get("geographic_risk_level", "low"),
            results.get("concentration_risk", {}).get("concentration_risk_level", "low")
        ]
        
        if "high" in risk_levels:
            results["overall_risk_level"] = "high"
        elif "medium" in risk_levels:
            results["overall_risk_level"] = "medium"
        else:
            results["overall_risk_level"] = "low"
        
        results["timestamp"] = datetime.now().isoformat()
        return results
    
    def get_active_risk_factors(self) -> Dict[str, Dict]:
        """
        Get the current active risk factors for all risk categories.
        
        Returns:
            Dictionary of active risk factors by category
        """
        return {
            "factor_exposures": self.factor_exposures.copy(),
            "sector_exposures": dict(self.sector_exposures),
            "geographic_exposures": dict(self.geographic_exposures)
        }
