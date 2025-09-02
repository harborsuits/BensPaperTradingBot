#!/usr/bin/env python3
"""
Forex Pair Manager

Handles loading and managing forex pair configurations from the forex_pairs.yaml file.
Provides access to pair properties, pip calculations, and optimal trading windows.
Integrates with BenBot for dynamic pair selection when available.
"""

import os
import yaml
import logging
import requests
import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forex_pair_manager')


@dataclass
class ForexPair:
    """Container for forex pair information."""
    symbol: str
    category: str
    base_currency: str
    quote_currency: str
    description: str
    pip_decimal: int
    pip_value_per_lot: float
    avg_spread_pips: float
    optimal_sessions: List[str]
    priority: int
    
    def get_pip_multiplier(self) -> float:
        """Get the multiplier to convert price differences to pips."""
        return 10 ** self.pip_decimal
    
    def price_to_pips(self, price_change: float) -> float:
        """Convert a price change to pips."""
        return price_change * self.get_pip_multiplier()
    
    def pips_to_price(self, pips: float) -> float:
        """Convert pips to a price change."""
        return pips / self.get_pip_multiplier()
    
    def calculate_pip_value(self, lot_size: float = 1.0) -> float:
        """Calculate the pip value for a given lot size."""
        return self.pip_value_per_lot * lot_size
    
    def calculate_spread_cost(self, lot_size: float = 1.0) -> float:
        """Calculate the cost of the spread in the account currency."""
        return self.avg_spread_pips * self.calculate_pip_value(lot_size)
    
    def is_jpy_pair(self) -> bool:
        """Check if this is a JPY pair."""
        return self.base_currency == 'JPY' or self.quote_currency == 'JPY'
    
    def get_account_risk(self, stop_loss_pips: float, lot_size: float, account_currency: str) -> float:
        """
        Calculate account risk in account currency.
        
        Args:
            stop_loss_pips: Stop loss distance in pips
            lot_size: Trading position size in lots
            account_currency: Account base currency
            
        Returns:
            Risk amount in account currency
        """
        # Direct calculation if account currency matches the quote
        if account_currency == self.quote_currency:
            return stop_loss_pips * self.calculate_pip_value(lot_size)
        
        # For other currencies, would need conversion rates
        # This is a simplified implementation
        return stop_loss_pips * self.calculate_pip_value(lot_size)


class ForexPairManager:
    """
    Manages forex pair information, pip calculations, and pair selection.
    
    Features:
    - Loads pair configurations from yaml
    - Calculates pip values and spread costs
    - Identifies correlated pairs
    - Integrates with BenBot for dynamic pair selection
    - Optimizes pair selection based on session and performance
    """
    
    def __init__(self, config_path: str = "forex_pairs.yaml"):
        """
        Initialize the Forex Pair Manager.
        
        Args:
            config_path: Path to the forex pairs configuration file
        """
        self.config_path = config_path
        self.config = {}
        self.pairs = {}
        self.correlations = {}
        self.benbot_config = {}
        self.last_benbot_update = None
        
        self.load_config()
        logger.info(f"Forex Pair Manager initialized with {len(self.pairs)} pairs")
        
        # Try to connect to BenBot if integration is enabled
        if self.config.get('settings', {}).get('benbot_integration', False):
            self.benbot_available = self._check_benbot_connection()
            if self.benbot_available:
                logger.info("BenBot integration confirmed")
                self._update_pairs_from_benbot()
            else:
                logger.warning("BenBot integration unavailable - using local configuration")
                
    def load_config(self) -> None:
        """Load the forex pair configuration from yaml file."""
        if not os.path.exists(self.config_path):
            logger.error(f"Configuration file not found: {self.config_path}")
            return
        
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
                
            # Extract settings
            self.settings = self.config.get('settings', {})
            
            # Extract pair information
            pairs_config = self.config.get('pairs', {})
            for symbol, pair_data in pairs_config.items():
                self.pairs[symbol] = ForexPair(
                    symbol=symbol,
                    category=pair_data.get('category', 'unknown'),
                    base_currency=pair_data.get('base_currency', symbol[:3]),
                    quote_currency=pair_data.get('quote_currency', symbol[3:6]),
                    description=pair_data.get('description', symbol),
                    pip_decimal=pair_data.get('pip_decimal', 4),
                    pip_value_per_lot=pair_data.get('pip_value_per_lot', 10.0),
                    avg_spread_pips=pair_data.get('avg_spread_pips', 1.0),
                    optimal_sessions=pair_data.get('optimal_sessions', []),
                    priority=pair_data.get('priority', 5)
                )
            
            # Extract correlation data
            self.correlations = self.config.get('correlations', {})
            
            # Extract BenBot integration config
            self.benbot_config = self.config.get('benbot', {})
            
            logger.info(f"Loaded configuration with {len(self.pairs)} pairs")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def _check_benbot_connection(self) -> bool:
        """Check if BenBot integration is available."""
        if not self.benbot_config.get('endpoint'):
            return False
        
        try:
            # Simple ping to BenBot
            endpoint = self.benbot_config.get('endpoint').rstrip('/') + '/ping'
            response = requests.get(endpoint, timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _update_pairs_from_benbot(self) -> None:
        """Update pair information from BenBot if available."""
        if not self.benbot_available or not self.benbot_config.get('request_preferred_pairs'):
            return
        
        # Check if we need to update
        now = datetime.datetime.now()
        update_frequency = self.benbot_config.get('update_frequency_hours', 12)
        
        if (self.last_benbot_update and 
            (now - self.last_benbot_update).total_seconds() < update_frequency * 3600):
            return
        
        try:
            endpoint = self.benbot_config.get('endpoint').rstrip('/') + '/pairs'
            
            payload = {
                'source': 'EvoTrader',
                'module': 'PairManager',
                'current_pairs': list(self.pairs.keys()),
                'timestamp': now.isoformat()
            }
            
            response = requests.post(endpoint, json=payload, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Update priorities based on BenBot's preferences
                if 'preferred_pairs' in data:
                    for pair_data in data['preferred_pairs']:
                        symbol = pair_data.get('symbol')
                        priority = pair_data.get('priority')
                        
                        if symbol in self.pairs and priority is not None:
                            self.pairs[symbol].priority = priority
                
                self.last_benbot_update = now
                logger.info("Updated pair priorities from BenBot")
                
        except Exception as e:
            logger.error(f"Error updating from BenBot: {e}")
    
    def get_pair(self, symbol: str) -> Optional[ForexPair]:
        """
        Get information for a specific forex pair.
        
        Args:
            symbol: The forex pair symbol (e.g., 'EURUSD')
            
        Returns:
            ForexPair object or None if not found
        """
        return self.pairs.get(symbol.upper())
    
    def get_correlation(self, pair1: str, pair2: str) -> float:
        """
        Get the correlation between two forex pairs.
        
        Args:
            pair1: First pair symbol
            pair2: Second pair symbol
            
        Returns:
            Correlation coefficient (-1.0 to 1.0) or 0.0 if unknown
        """
        pair1 = pair1.upper()
        pair2 = pair2.upper()
        
        # Check direct correlation
        if pair1 in self.correlations and pair2 in self.correlations[pair1]:
            return self.correlations[pair1][pair2]
        
        # Check reverse correlation
        if pair2 in self.correlations and pair1 in self.correlations[pair2]:
            return self.correlations[pair2][pair1]
        
        # No correlation data available
        return 0.0
    
    def get_pairs_by_category(self, category: str) -> List[ForexPair]:
        """
        Get all pairs in a specific category.
        
        Args:
            category: Category name (e.g., 'major', 'minor', 'exotic')
            
        Returns:
            List of ForexPair objects
        """
        return [pair for pair in self.pairs.values() if pair.category == category]
    
    def get_pairs_by_currency(self, currency: str) -> List[ForexPair]:
        """
        Get all pairs containing a specific currency.
        
        Args:
            currency: Currency code (e.g., 'USD', 'EUR')
            
        Returns:
            List of ForexPair objects
        """
        currency = currency.upper()
        return [
            pair for pair in self.pairs.values() 
            if pair.base_currency == currency or pair.quote_currency == currency
        ]
    
    def get_pairs_by_session(self, session: str) -> List[ForexPair]:
        """
        Get pairs with optimal trading during a specific session.
        
        Args:
            session: Session name (e.g., 'London', 'NewYork', 'Tokyo')
            
        Returns:
            List of ForexPair objects
        """
        return [
            pair for pair in self.pairs.values()
            if session in pair.optimal_sessions
        ]
    
    def get_priority_pairs(self, limit: int = 10) -> List[ForexPair]:
        """
        Get pairs sorted by priority (highest first).
        
        Args:
            limit: Maximum number of pairs to return
            
        Returns:
            List of ForexPair objects
        """
        # Ensure we have latest priorities if BenBot is available
        self._update_pairs_from_benbot()
        
        # Sort pairs by priority and return top N
        sorted_pairs = sorted(self.pairs.values(), key=lambda p: p.priority, reverse=True)
        return sorted_pairs[:limit]
    
    def get_uncorrelated_pairs(self, threshold: float = None) -> List[List[ForexPair]]:
        """
        Group pairs into uncorrelated clusters.
        
        Args:
            threshold: Correlation threshold (default: from config)
            
        Returns:
            List of lists, each containing uncorrelated pairs
        """
        if threshold is None:
            threshold = self.settings.get('correlation_threshold', 0.85)
        
        # Start with highest priority pair
        priority_pairs = self.get_priority_pairs(limit=len(self.pairs))
        
        # Build clusters
        clusters = []
        remaining = set(pair.symbol for pair in priority_pairs)
        
        while remaining:
            # Start a new cluster with highest priority remaining pair
            seed = next(pair for pair in priority_pairs if pair.symbol in remaining)
            cluster = [seed]
            remaining.remove(seed.symbol)
            
            # Find uncorrelated pairs to add to this cluster
            for pair in priority_pairs:
                if pair.symbol not in remaining:
                    continue
                    
                # Check correlation with all pairs in the cluster
                is_correlated = False
                for existing in cluster:
                    if abs(self.get_correlation(pair.symbol, existing.symbol)) > threshold:
                        is_correlated = True
                        break
                
                if not is_correlated:
                    cluster.append(pair)
                    remaining.remove(pair.symbol)
            
            clusters.append(cluster)
        
        return clusters
    
    def calculate_position_size(self, 
                              pair: str, 
                              risk_amount: float, 
                              stop_loss_pips: float,
                              account_currency: str = None) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            pair: Forex pair symbol
            risk_amount: Amount to risk in account currency
            stop_loss_pips: Stop loss distance in pips
            account_currency: Account currency (default: from config)
            
        Returns:
            Position size in lots
        """
        if account_currency is None:
            account_currency = self.settings.get('default_account_currency', 'USD')
            
        pair_obj = self.get_pair(pair)
        if not pair_obj:
            logger.warning(f"Pair not found: {pair}")
            return 0.0
            
        # Calculate pip value for 1 lot
        pip_value = pair_obj.calculate_pip_value(1.0)
        
        # Calculate lot size based on risk
        if pip_value > 0 and stop_loss_pips > 0:
            return risk_amount / (stop_loss_pips * pip_value)
        else:
            return 0.0
    
    def report_performance_to_benbot(self, 
                                   pair: str, 
                                   win_rate: float, 
                                   profit_factor: float,
                                   avg_pips: float,
                                   trades: int) -> bool:
        """
        Report pair performance metrics to BenBot.
        
        Args:
            pair: Forex pair symbol
            win_rate: Win rate percentage
            profit_factor: Profit factor
            avg_pips: Average pips per trade
            trades: Number of trades
            
        Returns:
            True if successfully reported, False otherwise
        """
        if not self.benbot_available or not self.benbot_config.get('report_performance'):
            return False
            
        try:
            endpoint = self.benbot_config.get('endpoint').rstrip('/') + '/performance'
            
            payload = {
                'source': 'EvoTrader',
                'module': 'PairManager',
                'pair': pair.upper(),
                'metrics': {
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'avg_pips': avg_pips,
                    'trades': trades,
                    'timestamp': datetime.datetime.now().isoformat()
                }
            }
            
            response = requests.post(endpoint, json=payload, timeout=5)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error reporting to BenBot: {e}")
            return False
    
    def create_pair_universe_report(self) -> str:
        """
        Generate a text report of the forex pair universe.
        
        Returns:
            Formatted string with pair information
        """
        report = "=== FOREX PAIR UNIVERSE ===\n\n"
        
        # Categorize pairs
        categories = {}
        for pair in self.pairs.values():
            if pair.category not in categories:
                categories[pair.category] = []
            categories[pair.category].append(pair)
        
        # Generate report by category
        for category, pairs in categories.items():
            cat_info = self.config.get('categories', {}).get(category, {})
            report += f"== {category.upper()} PAIRS ==\n"
            if 'description' in cat_info:
                report += f"{cat_info['description']}\n"
            
            report += "\n"
            
            # Sort pairs by priority
            sorted_pairs = sorted(pairs, key=lambda p: p.priority, reverse=True)
            
            for pair in sorted_pairs:
                report += f"{pair.symbol}: {pair.description}\n"
                report += f"  Priority: {pair.priority}\n"
                report += f"  Pip Value: ${pair.pip_value_per_lot:.2f} per lot\n"
                report += f"  Avg Spread: {pair.avg_spread_pips:.1f} pips\n"
                if pair.optimal_sessions:
                    report += f"  Optimal Sessions: {', '.join(pair.optimal_sessions)}\n"
                report += "\n"
            
            report += "\n"
        
        # Add correlation information
        report += "== NOTABLE CORRELATIONS ==\n\n"
        reported_correlations = set()
        
        for pair1 in self.pairs:
            if pair1 not in self.correlations:
                continue
                
            for pair2, corr in self.correlations[pair1].items():
                # Skip if already reported the reverse
                pair_key = tuple(sorted([pair1, pair2]))
                if pair_key in reported_correlations:
                    continue
                    
                # Only report strong correlations
                if abs(corr) >= 0.7:
                    report += f"{pair1}-{pair2}: {corr:.2f}\n"
                    reported_correlations.add(pair_key)
        
        return report


# Module execution
if __name__ == "__main__":
    import argparse
    from tabulate import tabulate
    
    parser = argparse.ArgumentParser(description="Forex Pair Manager")
    
    parser.add_argument(
        "--config", 
        type=str,
        default="forex_pairs.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--info", 
        type=str,
        help="Get detailed information for a specific pair (e.g., EURUSD)"
    )
    
    parser.add_argument(
        "--category", 
        type=str,
        choices=["major", "minor", "exotic"],
        help="List pairs in a specific category"
    )
    
    parser.add_argument(
        "--session", 
        type=str,
        help="List pairs optimal for a specific session"
    )
    
    parser.add_argument(
        "--uncorrelated", 
        action="store_true",
        help="Show uncorrelated pair groups"
    )
    
    parser.add_argument(
        "--calculate", 
        type=str,
        help="Calculate position size for a pair (specify in format PAIR:RISK:SL, e.g., EURUSD:100:20)"
    )
    
    parser.add_argument(
        "--report", 
        action="store_true",
        help="Generate full pair universe report"
    )
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = ForexPairManager(args.config)
    
    # Show info for a specific pair
    if args.info:
        pair = manager.get_pair(args.info.upper())
        if pair:
            print(f"\nInformation for {pair.symbol}:")
            print(f"Description: {pair.description}")
            print(f"Category: {pair.category}")
            print(f"Base/Quote: {pair.base_currency}/{pair.quote_currency}")
            print(f"Pip Decimal: {pair.pip_decimal}")
            print(f"Pip Value: ${pair.pip_value_per_lot:.2f} per standard lot")
            print(f"Avg Spread: {pair.avg_spread_pips:.1f} pips")
            print(f"Optimal Sessions: {', '.join(pair.optimal_sessions)}")
            print(f"Priority: {pair.priority}")
            
            # Show correlations
            print("\nCorrelations:")
            corr_data = []
            for other_symbol in manager.pairs:
                if other_symbol != pair.symbol:
                    corr = manager.get_correlation(pair.symbol, other_symbol)
                    if abs(corr) > 0:
                        corr_data.append([other_symbol, f"{corr:.2f}"])
            
            if corr_data:
                print(tabulate(sorted(corr_data, key=lambda x: abs(float(x[1])), reverse=True), 
                              headers=["Pair", "Correlation"], 
                              tablefmt="simple"))
            else:
                print("No correlation data available")
        else:
            print(f"Pair not found: {args.info}")
    
    # List pairs by category
    if args.category:
        pairs = manager.get_pairs_by_category(args.category)
        if pairs:
            print(f"\nPairs in category '{args.category}':")
            table_data = [
                [p.symbol, p.description, f"{p.avg_spread_pips:.1f}", p.priority]
                for p in sorted(pairs, key=lambda x: x.priority, reverse=True)
            ]
            print(tabulate(table_data, 
                          headers=["Symbol", "Description", "Avg Spread", "Priority"], 
                          tablefmt="simple"))
        else:
            print(f"No pairs found in category: {args.category}")
    
    # List pairs by session
    if args.session:
        pairs = manager.get_pairs_by_session(args.session)
        if pairs:
            print(f"\nPairs optimal for '{args.session}' session:")
            table_data = [
                [p.symbol, p.description, p.category, p.priority]
                for p in sorted(pairs, key=lambda x: x.priority, reverse=True)
            ]
            print(tabulate(table_data, 
                          headers=["Symbol", "Description", "Category", "Priority"], 
                          tablefmt="simple"))
        else:
            print(f"No pairs found for session: {args.session}")
    
    # Show uncorrelated pair groups
    if args.uncorrelated:
        groups = manager.get_uncorrelated_pairs()
        print("\nUncorrelated Pair Groups:")
        for i, group in enumerate(groups, 1):
            pairs_str = ", ".join(p.symbol for p in group)
            print(f"Group {i}: {pairs_str}")
    
    # Calculate position size
    if args.calculate:
        try:
            pair, risk, sl = args.calculate.split(":")
            risk_amount = float(risk)
            stop_loss_pips = float(sl)
            
            lot_size = manager.calculate_position_size(
                pair, risk_amount, stop_loss_pips
            )
            
            print(f"\nPosition Size Calculation for {pair}:")
            print(f"Risk Amount: ${risk_amount:.2f}")
            print(f"Stop Loss: {stop_loss_pips:.1f} pips")
            print(f"Position Size: {lot_size:.2f} lots")
            
            # Show additional information
            pair_obj = manager.get_pair(pair)
            if pair_obj:
                pip_value = pair_obj.calculate_pip_value(lot_size)
                spread_cost = pair_obj.calculate_spread_cost(lot_size)
                
                print(f"Pip Value for {lot_size:.2f} lots: ${pip_value:.2f}")
                print(f"Spread Cost: ${spread_cost:.2f}")
                print(f"Risk:Reward 1:{1.5}:")
                print(f"  Take Profit: {stop_loss_pips * 1.5:.1f} pips (${pip_value * 1.5:.2f})")
                print(f"  Stop Loss: {stop_loss_pips:.1f} pips (${pip_value:.2f})")
                
        except ValueError:
            print("Invalid format. Use PAIR:RISK:SL (e.g., EURUSD:100:20)")
    
    # Generate full report
    if args.report:
        report = manager.create_pair_universe_report()
        print(report)
        
    # Default behavior if no arguments provided
    if not (args.info or args.category or args.session or args.uncorrelated or args.calculate or args.report):
        # Show priority pairs
        priority_pairs = manager.get_priority_pairs()
        print("\nPriority Forex Pairs:")
        table_data = [
            [p.symbol, p.description, p.category, f"{p.avg_spread_pips:.1f}", p.priority]
            for p in priority_pairs
        ]
        print(tabulate(table_data, 
                      headers=["Symbol", "Description", "Category", "Avg Spread", "Priority"], 
                      tablefmt="simple"))
        
        print("\nUse --help to see available commands")
